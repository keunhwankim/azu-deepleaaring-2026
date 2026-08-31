# -*- coding: utf-8 -*-
"""1단계 추론: 순정 Qwen2.5-3B-Instruct + Majority Voting (LB 0.78339)

infer_baseline.py(파싱/프롬프트)와 infer_mv.py(MV 추론)를 하나로 합친 것입니다.
파일이 둘로 나뉘어 있어 혼동이 있었고, infer_baseline.py에는 더 이상 쓰지 않는
QLoRA 설정이 남아 있었습니다.

최종 제출은 이 파일 + apply_verifier.py의 2단계 구성입니다.
  1단계 (이 파일)  : MV8 생성 -> submission_..._lb_*.csv   (LB 0.78339, 안전선)
  2단계 (검증기)   : 4~5표 구간 재선별 -> LB 0.78700

8/29 추가 (본시험 2,000문제 대응):
  - MV_INPUT 환경변수로 입력 CSV 경로를 지정 가능 (기존엔 하드코딩)
  - 제출 파일의 ID 컬럼명을 소문자 'id'로 강제 (규칙 6)
  - 생성 전에 프롬프트 토큰 길이를 점검 (max_model_len 초과 사전 경고)

실행:
  # 본시험
  MV_TARGET=lb MV_INPUT=/workspace/test.csv MV_DATE=0831 python infer_mv_final.py
  # LB 831 재현
  MV_TARGET=lb MV_DATE=0831 python infer_mv_final.py

주요 환경변수:
  MV_TARGET   lb / val / train      (기본 val)
  MV_INPUT    입력 CSV 경로          (lb일 때만, 기본 leaderboard_filtered)
  MV_DATE     파일명에 박을 날짜
  MV_N        샘플 수 (기본 8)
  MV_TEMP     온도 (기본 0.7)
  MV_LORA     어댑터 경로 (미지정 시 순정 — 최종안은 순정)
"""

import os, sys, re, json
from collections import Counter
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# ============================================================
# 1. 상수 — 다른 스크립트가 import해서 씁니다
#    (apply_verifier.py, train_lora.py, make_*.py, analyze_errors.py)
# ============================================================
MODEL_NAME     = "Qwen/Qwen2.5-3B-Instruct"
SEED           = 42
MAX_NEW_TOKENS = 2048     # 생성 상한. 실측 잘림률 0.38%
MAX_MODEL_LEN  = 3072     # 프롬프트 + 생성의 합

# 정답으로 인정할 절댓값 상한.
# 대회 정답은 정수이고 천문학적 값이 나올 리 없으므로, 이 범위를 넘으면
# "정규식이 엉뚱한 숫자열을 집었다"고 보는 게 타당합니다.
# (pandas가 초거대 int를 처리하다 OverflowError로 죽는 것도 함께 막습니다)
MAX_ABS_ANSWER = 10 ** 15

# 8/6에 정한 원본. 절대 수정하지 마십시오.
# 검증기 학습 데이터와 D단계 프롬프트 실험이 모두 이 문자열을 기준으로 합니다.
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step. "
    "Explain your reasoning clearly and logically. On the final line, output "
    "your answer in exactly this format: 'Answer: N' where N is an integer, "
    "with nothing else after it."
)

# ============================================================
# 2. 정답 파싱
# ============================================================
# "모델이 최종 답을 냈다"고 인정할 형식들. 위에서부터 우선순위가 높습니다.
#   - \boxed{}를 1순위로 둔 이유: Qwen 계열은 사전학습 습관상 최종 답을
#     \boxed{}에 넣는 경우가 많고, 풀이 중간에 'answer'라는 단어가
#     섞여 나오더라도 진짜 최종 답은 \boxed{} 안에 있을 확률이 높습니다.
#   - \}? 부분은 LaTeX \text{Answer: } 1862 같은 케이스를 잡기 위한 것입니다.
ANSWER_PATTERNS = [
    r"\\boxed\s*\{\s*(-?[\d,]+)\s*\}",
    r"[Aa]nswer\s*[:：]?\s*\}?\s*(-?[\d,]+)",
    r"[Tt]he answer is\s*\$?\s*(-?[\d,]+)",
]


def _safe_int(s: str):
    """문자열을 정수로 변환. 실패하거나 상식 범위를 벗어나면 None."""
    try:
        v = int(s.replace(",", ""))
    except (ValueError, TypeError):
        return None
    return v if abs(v) <= MAX_ABS_ANSWER else None


def extract_formatted(text: str):
    """지정된 형식으로 답을 냈으면 그 정수를 반환, 아니면 None."""
    for pat in ANSWER_PATTERNS:
        found = re.findall(pat, text)
        # 여러 번 등장하면 뒤에서부터 확인 (최종 결론일 가능성이 높음)
        for cand in reversed(found):
            v = _safe_int(cand)
            if v is not None:
                return v
    return None


def parse_answer(text: str):
    """생성문에서 최종 정수 정답을 뽑아냅니다.
    반환: (정답, 어느 단계에서 건졌는지)
    빈칸을 남기면 채점 시 에러가 날 수 있어 어떤 경우에도 값을 냅니다."""
    text = str(text)

    val = extract_formatted(text)            # 1순위: 지정된 형식
    if val is not None:
        return val, "format_ok"

    # 2순위: 형식 미준수 시 뒤에서부터 훑어 정상 범위의 첫 정수
    #        (풀이 중간 계산값을 집을 위험이 있어 신뢰도가 낮습니다)
    for s in reversed(re.findall(r"-?\d[\d,]*", text)):
        v = _safe_int(s)
        if v is not None:
            return v, "fallback_lastnum"

    return 0, "failed_zero"                  # 3순위: 쓸 만한 숫자 없음


# ============================================================
# 3. 시스템 프롬프트 후보 (D단계 실험용, 최종안은 P0)
# ============================================================
# 원본을 두 조각으로 잘라 재사용합니다. 직접 타이핑하면 오타 한 글자로
# 대조군이 무효가 되므로, 반드시 잘라서 씁니다.
_CUT_FMT  = "On the final line,"
_CUT_HEAD = "Explain your reasoning"
assert _CUT_FMT in SYSTEM_PROMPT and _CUT_HEAD in SYSTEM_PROMPT

_FMT  = SYSTEM_PROMPT[SYSTEM_PROMPT.index(_CUT_FMT):]     # 형식 지시 (모든 arm 공유)
_HEAD = SYSTEM_PROMPT[:SYSTEM_PROMPT.index(_CUT_FMT)]     # 역할 + "Explain your reasoning..."
_ROLE = SYSTEM_PROMPT[:SYSTEM_PROMPT.index(_CUT_HEAD)]    # 역할 문장만

PROMPTS = {
    "P0":  SYSTEM_PROMPT,   # ★ 최종 채택. 원본 그대로
    "P0b": SYSTEM_PROMPT,   # 대조군 재실행용 (노이즈 측정)
}

# 아래 넷은 전부 LB에서 기각되었습니다. 실험 재현용으로만 남깁니다.
PROMPTS["P1units"] = _HEAD + (          # 배율/단위 실수 겨냥 (기각)
    "Be careful with scale and units: watch for doubling, halving, factors of "
    "10 or 100, and unit conversions (cm/m, minutes/hours). Before giving the "
    "final answer, re-read the question and confirm your number is the exact "
    "quantity it asks for, in the unit it asks for."
) + " " + _FMT

PROMPTS["P2contra"] = _HEAD + (         # 모순 감지 시 복귀 (LB 0.77135)
    "As you go, check each intermediate result against the problem's stated "
    "conditions. If a result contradicts a condition or one of your own earlier "
    "steps, say so explicitly, go back to that step, and redo it instead of "
    "continuing forward."
) + " " + _FMT

PROMPTS["P3cond"] = _ROLE + (           # 조건 선열거 (기각)
    "First, list every condition the problem gives and the exact quantity it "
    "asks for, as a short bulleted list. Then solve step by step, explaining "
    "your reasoning clearly and logically. Before finishing, confirm that every "
    "listed condition was used."
) + " " + _FMT

PROMPTS["P4check"] = _HEAD + (          # 역대입 검산 (LB 0.76052, 최악)
    "Once you reach a candidate answer, verify it by substituting it back into "
    "the problem's conditions and checking that all of them hold. If the check "
    "fails, solve the problem again."
) + " " + _FMT

PROMPTS["P5nocheck"] = _HEAD + (        # 재확인 억제 (LB 0.77617)
    "Solve the problem in one pass. Do not go back to re-check or revise "
    "your earlier steps. If you are unsure, give your best estimate and "
    "move on."
) + " " + _FMT


# ============================================================
# 4. 메인 — import 시 실행되지 않도록 함수로 감쌉니다
#    (apply_verifier.py 등이 위 상수/함수만 가져다 씁니다)
# ============================================================
def main():
    N_SAMPLES   = int(os.environ.get("MV_N", 8))
    TEMPERATURE = float(os.environ.get("MV_TEMP", 0.7))   # 0이면 투표가 무의미
    TOP_P       = 0.95
    LORA_PATH   = os.environ.get("MV_LORA") or None       # 미지정 시 순정 (최종안)
    LORA_RANK   = int(os.environ.get("MV_LORA_RANK", 16))

    PROMPT_ID = os.environ.get("MV_PROMPT", "P0")
    if PROMPT_ID not in PROMPTS:
        raise SystemExit(f"모르는 프롬프트 ID: {PROMPT_ID}\n  사용 가능: {list(PROMPTS)}")
    ACTIVE_PROMPT = PROMPTS[PROMPT_ID]

    TARGET   = os.environ.get("MV_TARGET", "val")
    DATE     = os.environ.get("MV_DATE", "0831")
    TAG      = "BASE" if LORA_PATH is None else os.environ.get("MV_TAG", "LORA")
    RUN_NAME = f"{TAG}_{PROMPT_ID}_mv{N_SAMPLES}_t{TEMPERATURE}_{TARGET}_{DATE}"
    OUT_DIR  = "/workspace/outputs"

    RAW_CSV = f"{OUT_DIR}/_raw_only_{RUN_NAME}.csv"
    MV_CSV  = f"{OUT_DIR}/mvresult_{RUN_NAME}.csv"
    SUB_CSV = f"{OUT_DIR}/submission_{RUN_NAME}.csv"

    # 덮어쓰기 방지 — GPU 시간을 쓰기 전에 미리 막습니다
    os.makedirs(OUT_DIR, exist_ok=True)
    for _p in (RAW_CSV, MV_CSV):
        if os.path.exists(_p):
            raise SystemExit(f"[중단] 이미 존재: {_p}\n  MV_DATE를 바꾸거나 지우세요.")

    print("=" * 70)
    print(f"[설정] target={TARGET}  n={N_SAMPLES}  temp={TEMPERATURE}  top_p={TOP_P}")
    print(f"       max_tokens={MAX_NEW_TOKENS}  max_model_len={MAX_MODEL_LEN}  seed={SEED}")
    print(f"       어댑터   : {LORA_PATH or '없음 (순정 baseline) ← 최종안'}")
    print(f"       프롬프트 : {PROMPT_ID}")
    print(f"       └ {ACTIVE_PROMPT}")
    print(f"       출력     : {RUN_NAME}")
    print("=" * 70)

    # ---------- 데이터 로드 ----------
    if TARGET == "val":
        qs, golds, ids = [], [], []
        with open(f"{OUT_DIR}/val_sft.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                qs.append(next(m["content"] for m in msgs if m["role"] == "user"))
                g, _ = parse_answer(next(m["content"] for m in msgs
                                         if m["role"] == "assistant"))
                golds.append(g); ids.append(i)
        id_col = "idx"

    elif TARGET == "train":
        # 검증기 학습 데이터 생성용. train 17,000문제 전량은 7시간 이상이라 샘플링합니다.
        # random_state 고정이므로 나중에 id로 원본과 다시 붙일 수 있습니다.
        df = pd.read_csv("/workspace/deep_chal_math_train.csv")
        n = int(os.environ.get("MV_SUBSAMPLE", 3000))
        if n < len(df):
            df = df.sample(n=n, random_state=42).reset_index(drop=True)
        id_col = "id"
        qs    = [str(q) for q in df["question"]]
        ids   = list(df["id"])
        golds = [int(a) for a in df["answer"]]

    else:   # lb / 본시험
        INPUT = os.environ.get("MV_INPUT",
                               "/workspace/deep_chal_math_leaderboard_filtered.csv")
        print(f"[입력] {INPUT}")
        df = pd.read_csv(INPUT)
        print(f"  실제 컬럼: {list(df.columns)}")
        id_col = next((c for c in df.columns if c.lower() in ("id", "problem_id")), None)
        q_col  = next((c for c in df.columns
                       if c.lower() in ("question", "problem", "text")), None)
        if id_col is None or q_col is None:
            raise SystemExit(
                f"ID/question 컬럼을 못 찾았습니다. 실제 컬럼: {list(df.columns)}")
        print(f"  사용: id='{id_col}', question='{q_col}'")
        qs, ids, golds = [str(q) for q in df[q_col]], list(df[id_col]), None

    print(f"[데이터] 문제 {len(qs)}개")

    # ---------- 프롬프트 + 길이 사전 점검 ----------
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": ACTIVE_PROMPT}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True) for q in qs]

    # 프롬프트가 길면 생성 여유가 없어 답을 못 냅니다. 미리 알아야 대응이 됩니다.
    plen = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    room = MAX_MODEL_LEN - max(plen)
    n_tight = sum(1 for L in plen if MAX_MODEL_LEN - L < 512)
    print(f"[길이] 프롬프트 최대 {max(plen)}토큰, 중앙값 {sorted(plen)[len(plen)//2]}")
    print(f"       가장 긴 문제의 생성 여유 {room}토큰"
          f"   생성 여유 512 미만 문제 {n_tight}개")
    if room <= 0:
        raise SystemExit("[중단] 프롬프트가 max_model_len을 초과합니다. "
                         "MAX_MODEL_LEN을 늘리거나 해당 문제를 별도 처리하세요.")
    if n_tight:
        print("       ↑ 이 문제들은 답까지 못 갈 수 있습니다 (fallback 처리됨)")

    # ---------- 생성 ----------
    llm = LLM(model=MODEL_NAME, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=MAX_MODEL_LEN, seed=SEED,
              enable_lora=LORA_PATH is not None, max_lora_rank=LORA_RANK)

    # n=N_SAMPLES: 프롬프트 하나당 N개 생성. vLLM이 배치로 묶어 처리해 효율적.
    outs = llm.generate(
        prompts,
        SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                       max_tokens=MAX_NEW_TOKENS, seed=SEED),
        lora_request=LoRARequest("adapter", 1, LORA_PATH) if LORA_PATH else None,
    )

    # ---------- 생성 원본 즉시 백업 ----------
    # 파싱/저장에서 죽더라도 GPU 결과는 살아남습니다. 재추론이 필요 없습니다.
    raw_all = [[c.text for c in o.outputs] for o in outs]
    meta_all = [[{"fr": getattr(c, "finish_reason", "unknown"), "nt": len(c.token_ids)}
                 for c in o.outputs] for o in outs]

    pd.DataFrame({
        id_col: ids,
        "raw_json":  [json.dumps(r) for r in raw_all],
        "meta_json": [json.dumps(m) for m in meta_all],
    }).to_csv(RAW_CSV, index=False)
    print(f"[백업] {RAW_CSV}")

    _flat  = [s for m in meta_all for s in m]
    _trunc = sum(1 for s in _flat if s["fr"] == "length")
    print(f"[진단] 잘림 {_trunc}/{len(_flat)} = {_trunc/len(_flat)*100:.2f}%"
          f"   평균 생성 {sum(s['nt'] for s in _flat)/len(_flat):.0f}토큰")

    # ---------- 투표 ----------
    # 집계 규칙: format_ok 샘플만 투표에 참여, 합의도 분모는 N 고정.
    # 오프라인에서 682/682 완전 일치로 검증된 규칙입니다.
    finals, vote_counts, levels_summary = [], [], []
    for samples in raw_all:
        cands, lvls = [], []
        for t in samples:
            v, l = parse_answer(t)
            lvls.append(l)
            # fallback(마지막 정수 줍기)은 신뢰도가 낮아 표를 오염시킵니다.
            if l == "format_ok":
                cands.append(v)
        if not cands:                  # 전부 형식 실패면 어쩔 수 없이 전체 사용
            cands = [parse_answer(t)[0] for t in samples]
        top, cnt = Counter(cands).most_common(1)[0]
        finals.append(top)
        vote_counts.append(cnt / len(samples))
        levels_summary.append(sum(1 for l in lvls if l == "format_ok") / len(lvls))

    # ---------- 결과 ----------
    res = pd.DataFrame({id_col: ids, "answer": finals,
                        "agreement": vote_counts, "format_rate": levels_summary})

    if golds is not None:
        res["gold"] = golds
        res["ok"] = res["answer"] == res["gold"]
        print("\n" + "=" * 60)
        print(f"  프롬프트     : {PROMPT_ID}   (target={TARGET})")
        print(f"  정답률       : {res['ok'].sum()}/{len(res)} = {res['ok'].mean():.4f}")
        print(f"  평균 합의도  : {res['agreement'].mean():.4f}")
        print(f"  평균 format율: {res['format_rate'].mean():.4f}")
        print(f"  만장일치 문제: {(res['agreement'] == 1.0).sum()}개 "
              f"(그중 정답률 {res[res['agreement'] == 1.0]['ok'].mean():.4f})")
        v = (res["agreement"] * N_SAMPLES).round().astype(int)
        print("  표수 분포    : " +
              "  ".join(f"{k}표:{(v==k).sum()}" for k in range(1, N_SAMPLES + 1)))
        print("=" * 60)
    else:
        # 규칙 6: 제출 컬럼은 소문자 'id'와 'answer'.
        # 입력 파일이 'ID'로 되어 있어도 여기서 강제로 맞춥니다.
        sub = res[[id_col, "answer"]].copy().rename(columns={id_col: "id"})
        sub["answer"] = sub["answer"].astype("int64")
        sub.to_csv(SUB_CSV, index=False)
        print(f"\n  제출 파일: {SUB_CSV}")
        print(f"  행 {len(sub)} / 컬럼 {list(sub.columns)} / 결측 {sub.isna().sum().sum()}")
        print(f"\n  2단계(검증기)를 이어서 실행하려면:")
        print(f"    VER_TARGET=lb VER_TAG=final \\")
        print(f"      VER_MV={MV_CSV} \\")
        print(f"      VER_RAW={RAW_CSV} \\")
        print(f"      VER_Q={os.environ.get('MV_INPUT', '<입력파일>')} \\")
        print(f"      python apply_verifier.py")

    res.to_csv(MV_CSV, index=False)
    print(f"  집계 결과: {MV_CSV}")


if __name__ == "__main__":
    main()
