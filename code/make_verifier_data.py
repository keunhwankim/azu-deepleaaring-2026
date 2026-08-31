# -*- coding: utf-8 -*-
"""검증기 학습 데이터 생성 (GPU 불필요, 약 1분)

train MV8 결과의 8샘플 각각에 대해 gold와 대조하여 Correct/Incorrect 라벨을 붙입니다.
STaR와 결정적으로 다른 점: 오답도 학습에 씁니다. 판별 경계는 오답이 있어야 배웁니다.

출력:
  verifier_train.jsonl   검증기 학습용 (messages 형식, train_lora.py가 그대로 읽음)
  holdout_ids.json       홀드아웃 문제 id 목록 (평가 전용, 학습에 미사용)

실행: python make_verifier_data.py
"""

import json, random
from collections import Counter
import pandas as pd

D        = "/workspace/outputs"
MV_CSV   = f"{D}/mvresult_BASE_P0_mv8_t0.7_train_0824.csv"
RAW_CSV  = f"{D}/_raw_only_BASE_P0_mv8_t0.7_train_0824.csv"
TRAIN_CSV = "/workspace/deep_chal_math_train.csv"
N        = 8
HOLDOUT  = 600          # 편향 없는 평가셋. LB를 val보다 잘 대변합니다
SEED     = 42
MAX_CHARS = 3000        # 풀이가 너무 길면 학습 시 잘려서 라벨이 날아감

# 검증기 전용 시스템 프롬프트. 생성용(SYSTEM_PROMPT)과 달라야 합니다.
VERIFIER_SYSTEM = (
    "You are a strict grader for math solutions. You will be shown a problem "
    "and one candidate solution. Judge whether the solution's final answer is "
    "correct. Reply with exactly one word: Correct or Incorrect."
)

random.seed(SEED)

# ---------- 1. 로드 ----------
mv  = pd.read_csv(MV_CSV)
raw = pd.read_csv(RAW_CSV)
tr  = pd.read_csv(TRAIN_CSV)[["id", "question"]]
df  = mv.merge(raw[["id", "raw_json"]], on="id").merge(tr, on="id")
df["votes"] = (df["agreement"] * N).round().astype(int)
print(f"[로드] {len(df)}문제")

# ---------- 2. 홀드아웃 분할 (라벨링 전에 먼저) ----------
# 먼저 나누지 않으면 학습 데이터가 평가셋에 새어 들어갑니다.
ids = list(df["id"])
random.shuffle(ids)
hold_ids  = set(ids[:HOLDOUT])
train_ids = set(ids[HOLDOUT:])
json.dump(sorted(hold_ids), open(f"{D}/holdout_ids.json", "w"))
print(f"[분할] 학습 {len(train_ids)} / 홀드아웃 {len(hold_ids)}")

# ---------- 3. 파싱 ----------
import sys
sys.path.insert(0, "/workspace")
from infer_baseline import parse_answer

# ---------- 4. 라벨링 ----------
# 만장일치(8표) 문제는 8샘플이 전부 정답이라 판별 경계를 못 가르칩니다.
# 다만 전부 빼면 "Correct" 사례가 부족해지므로 일부만 섞습니다.
UNANIMOUS_KEEP = 0.15

rows, stat = [], Counter()
for _, r in df.iterrows():
    if r["id"] not in train_ids:
        continue
    v = r["votes"]
    if v == 1:                                  # 1표 구간은 대부분 지식 문제라 제외
        stat["skip_1vote"] += 1
        continue
    if v == N and random.random() > UNANIMOUS_KEEP:
        stat["skip_unanimous"] += 1
        continue

    gold = int(r["gold"])
    for text in json.loads(r["raw_json"]):
        val, lvl = parse_answer(text)
        if lvl != "format_ok":                  # 형식 실패는 투표에도 안 들어감
            stat["skip_format"] += 1
            continue
        if len(text) > MAX_CHARS:
            stat["skip_long"] += 1
            continue
        label = "Correct" if val == gold else "Incorrect"
        stat[label] += 1
        rows.append({"messages": [
            {"role": "system",    "content": VERIFIER_SYSTEM},
            {"role": "user",      "content":
                f"Problem:\n{r['question']}\n\nCandidate solution:\n{text.strip()}"},
            {"role": "assistant", "content": label},
        ]})

# ---------- 5. 클래스 균형 ----------
# 한쪽이 압도적이면 모델이 그냥 다수 클래스만 뱉는 법을 배웁니다.
pos = [x for x in rows if x["messages"][-1]["content"] == "Correct"]
neg = [x for x in rows if x["messages"][-1]["content"] == "Incorrect"]
k = min(len(pos), len(neg))
rows = random.sample(pos, k) + random.sample(neg, k)
random.shuffle(rows)

with open(f"{D}/verifier_train.jsonl", "w", encoding="utf-8") as f:
    for x in rows:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print(f"\n[라벨] Correct {stat['Correct']} / Incorrect {stat['Incorrect']}")
print(f"       제외 — 1표 {stat['skip_1vote']}문제, 만장일치 {stat['skip_unanimous']}문제, "
      f"형식실패 {stat['skip_format']}샘플, 과길이 {stat['skip_long']}샘플")
print(f"[균형] 각 {k}개씩 -> 총 {len(rows)}개")
print(f"[저장] {D}/verifier_train.jsonl")

# ---------- 6. 홀드아웃 상한 (검증기가 아무리 잘해도 넘을 수 없는 선) ----------
h = df[df["id"].isin(hold_ids)]
seg = h[h["votes"].isin([4, 5])]
n_ok = n_ceil = 0
for _, r in seg.iterrows():
    gold = int(r["gold"])
    cands = [v for v, l in (parse_answer(t) for t in json.loads(r["raw_json"]))
             if l == "format_ok"]
    n_ok   += (r["answer"] == gold)
    n_ceil += (gold in cands)
print(f"\n{'=' * 60}")
print(f"홀드아웃 {len(h)}문제 중 4~5표 구간: {len(seg)}문제")
print(f"  현재 MV : {n_ok:3d}  ({n_ok/max(len(seg),1):.3f})")
print(f"  상한    : {n_ceil:3d}  ({n_ceil/max(len(seg),1):.3f})")
print(f"  회수 여지: {n_ceil - n_ok}문제")
print("=" * 60)
print("※ 회수 여지가 10문제 미만이면 검증기가 완벽해도 노이즈를 못 넘습니다.")
