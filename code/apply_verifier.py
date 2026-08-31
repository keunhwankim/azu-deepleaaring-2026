# -*- coding: utf-8 -*-
"""검증기 어댑터로 4~5표 구간을 재선별.

동작:
  1. 대상 raw의 4~5표 구간 문제만 골라, 8샘플 각각을 검증기로 채점
  2. 검증기 출력 첫 토큰의 logprob에서 P(Correct)를 계산
  3. 답 후보별로 점수를 합산해 최다 점수 답을 채택
  4. 4~5표 이외 구간은 기존 MV 결과를 그대로 둠 (0.95+ 구간을 망치지 않기 위해)

가중치를 "합"으로 쓰는 이유:
  평균을 쓰면 표본이 1개인 소수 후보가 구조적으로 유리해집니다.
  로그확률 진단에서 이것 때문에 정확도가 0.760 -> 0.385로 붕괴했습니다.
  합을 쓰면 표 수와 확신도가 함께 반영됩니다.

실행:
  VER_TARGET=holdout python apply_verifier.py     # 홀드아웃 600문제로 판정
  VER_TARGET=lb       python apply_verifier.py     # LB 제출본 생성
"""

import os, sys, json, math
from collections import defaultdict
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace")
from infer_mv_final import parse_answer, MODEL_NAME

D          = "/workspace/outputs"
VER_PATH   = "/workspace/checkpoints/verifier_r16/final"
TARGET     = os.environ.get("VER_TARGET", "holdout")
TAG        = os.environ.get("VER_TAG", "v1")
N          = int(os.environ.get("VER_N", 8))
LO = int(os.environ.get("VER_LO", 4))
MARGIN = float(os.environ.get("VER_MARGIN", 0.0))
HI = int(os.environ.get("VER_HI", 5))              # 적용 구간. 이 밖은 기존 MV 결과 유지
MAXLEN     = 2048
TOK_OK, TOK_NG = 33092, 40468  # 'Correct', 'Incorrect' 토큰 id

VERIFIER_SYSTEM = (
    "You are a strict grader for math solutions. You will be shown a problem "
    "and one candidate solution. Judge whether the solution's final answer is "
    "correct. Reply with exactly one word: Correct or Incorrect."
)

# ---------- 1. 대상 로드 ----------
if TARGET == "holdout":
    mv   = pd.read_csv(f"{D}/mvresult_BASE_P0_mv8_t0.7_train_0824.csv")
    raw  = pd.read_csv(f"{D}/_raw_only_BASE_P0_mv8_t0.7_train_0824.csv")
    qsrc = pd.read_csv("/workspace/deep_chal_math_train.csv")[["id", "question"]]
    keep = set(json.load(open(f"{D}/holdout_ids.json")))
    df   = mv.merge(raw[["id", "raw_json"]], on="id").merge(qsrc, on="id")
    df   = df[df["id"].isin(keep)].reset_index(drop=True)   # 학습에 안 쓴 문제만
    id_col, has_gold = "id", True
else:  # lb
    # 8/31 본 시험용: 환경변수로 경로를 넘길 수 있게 함 (기본값은 LB 831 재현)
    mv   = pd.read_csv(os.environ.get("VER_MV",  f"{D}/BASE_mv8_lb_0813_mv.csv"))
    raw  = pd.read_csv(os.environ.get("VER_RAW", f"{D}/BASE_mv8_lb_0813_raw.csv"))
    qsrc = pd.read_csv(os.environ.get("VER_Q",
        "/workspace/deep_chal_math_leaderboard_filtered.csv"))
    qcol = next(c for c in qsrc.columns if c.lower() in ("question","problem","text"))
    icol = next(c for c in qsrc.columns if c.lower() in ("id","problem_id"))
    qsrc = qsrc[[icol, qcol]].rename(columns={icol: "id", qcol: "question"})
    key  = "id" if "id" in mv.columns else mv.columns[0]
    df   = mv.merge(raw[[key, "raw_json"]], on=key).merge(qsrc, left_on=key, right_on="id")
    id_col, has_gold = key, False

df["votes"] = (df["agreement"] * N).round().astype(int)
seg = df[df["votes"].between(LO, HI)].reset_index(drop=True)
print(f"[대상] {TARGET}: 전체 {len(df)}문제, {LO}~{HI}표 구간 {len(seg)}문제")

# ---------- 2. 채점 프롬프트 ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
prompts, meta = [], []          # meta: (seg 행 인덱스, 파싱된 답)
for i, r in seg.iterrows():
    for text in json.loads(r["raw_json"]):
        v, lvl = parse_answer(text)
        if lvl != "format_ok":          # 원래 투표에도 안 들어간 샘플
            continue
        p = tok.apply_chat_template(
            [{"role": "system", "content": VERIFIER_SYSTEM},
             {"role": "user",   "content":
                 f"Problem:\n{r['question']}\n\nCandidate solution:\n{str(text).strip()}"}],
            tokenize=False, add_generation_prompt=True)
        if len(tok(p, add_special_tokens=False)["input_ids"]) >= MAXLEN - 8:
            continue
        prompts.append(p); meta.append((i, v))
print(f"[채점] {len(prompts)}샘플")

# ---------- 3. 검증기 실행 ----------
llm = LLM(model=MODEL_NAME, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=MAXLEN, seed=42, enable_lora=True, max_lora_rank=16)
outs = llm.generate(
    prompts,
    SamplingParams(max_tokens=1, temperature=0.0, logprobs=20),
    lora_request=LoRARequest("verifier", 1, VER_PATH),
)

# ---------- 4. P(Correct) ----------
# 첫 토큰의 logprob에서 두 후보만 뽑아 정규화합니다.
scores, missing = [], 0
for o in outs:
    lp = o.outputs[0].logprobs[0]
    a = lp[TOK_OK].logprob if TOK_OK in lp else -20.0
    b = lp[TOK_NG].logprob if TOK_NG in lp else -20.0
    if TOK_OK not in lp and TOK_NG not in lp:
        missing += 1
    scores.append(math.exp(a) / (math.exp(a) + math.exp(b)))
scores = np.array(scores)
print(f"[점수] 평균 P(Correct) {scores.mean():.4f}  "
      f"중앙값 {np.median(scores):.4f}  두 토큰 모두 없음 {missing}")

# ---------- 5. 가중 투표 ----------
agg = defaultdict(lambda: defaultdict(float))   # seg행 -> 답 -> 점수합
cnt = defaultdict(lambda: defaultdict(int))
for (i, v), s in zip(meta, scores):
    agg[i][v] += s
    cnt[i][v] += 1

new_answer = {}
for i in range(len(seg)):
    if not agg[i]:
        continue
    ranked = sorted(agg[i].items(), key=lambda kv: -kv[1])
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    # 1·2위 점수차가 MARGIN 미만이면 교체하지 않음 (홀드아웃: 개악 4 -> 1)
    if ranked[0][1] - second >= MARGIN:
        new_answer[i] = ranked[0][0]

seg["new_answer"] = [new_answer.get(i, seg.loc[i, "answer"]) for i in range(len(seg))]
changed = int((seg["new_answer"] != seg["answer"]).sum())
print(f"[변경] {changed}/{len(seg)}문제의 답이 바뀜")

# ---------- 6. 판정 또는 제출 ----------
if has_gold:
    g = seg["gold"].astype(int)
    old = int((seg["answer"] == g).sum())
    new = int((seg["new_answer"] == g).sum())
    ceil_ = sum(int(seg.loc[i, "gold"]) in agg[i] for i in range(len(seg)) if agg[i])
    # 어느 방향으로 바뀌었는지
    imp = int(((seg["answer"] != g) & (seg["new_answer"] == g)).sum())
    dmg = int(((seg["answer"] == g) & (seg["new_answer"] != g)).sum())
    print("\n" + "=" * 60)
    print(f"  {LO}~{HI}표 구간 {len(seg)}문제")
    print(f"  기존 MV : {old:3d}  ({old/len(seg):.3f})")
    print(f"  검증기  : {new:3d}  ({new/len(seg):.3f})   순이득 {new-old:+d}")
    print(f"            개선 +{imp} / 개악 -{dmg}")
    print(f"  상한    : {ceil_:3d}  ({ceil_/len(seg):.3f})")
    print(f"\n  전체 {len(df)}문제 기준: "
          f"{int((df['answer']==df['gold']).sum())} -> "
          f"{int((df['answer']==df['gold']).sum()) + (new-old)}")
    print("=" * 60)
    print("※ 홀드아웃 600문제 노이즈는 약 +-10. 순이득이 +10 미만이면 기각.")
else:
    out = df.copy()
    m = dict(zip(seg[id_col], seg["new_answer"]))
    out["answer"] = [m.get(i, a) for i, a in zip(out[id_col], out["answer"])]
    sub = out[[id_col, "answer"]].rename(columns={id_col: "id"})
    sub["answer"] = sub["answer"].astype("int64")
    p = f"{D}/submission_VERIFIER_mv8_lb_0825.csv"
    sub.to_csv(p, index=False)
    print(f"\n  제출 파일: {p}")
    print(f"  행 {len(sub)} / 컬럼 {list(sub.columns)} / 결측 {sub.isna().sum().sum()}")

seg.to_csv(f"{D}/verifier_seg_{TARGET}_{TAG}.csv", index=False)

# 샘플별 점수 저장 — 이게 있어야 결합 방식을 GPU 없이 다시 실험할 수 있습니다
pd.DataFrame({
    "seg_row": [i for i, _ in meta],
    "id":      [seg.loc[i, id_col] for i, _ in meta],
    "answer":  [v for _, v in meta],
    "p_correct": scores,
}).to_csv(f"{D}/verifier_scores_{TARGET}_{TAG}.csv", index=False)
print(f"  샘플 점수: {D}/verifier_scores_{TARGET}_{TAG}.csv")
