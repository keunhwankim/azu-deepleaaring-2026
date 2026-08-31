#!/bin/bash
# 사용법: bash run_all.sh <test.csv> [출력디렉터리]
set -e
TEST="${1:?사용법: bash run_all.sh <test.csv> [outdir]}"
OUT="${2:-./outputs}"
mkdir -p "$OUT"
export OUT_DIR="$OUT"
D=$(date +%m%d%H%M)

echo "=== 1단계: MV8 생성 (약 60분/2000문제) ==="
MV_TARGET=lb MV_INPUT="$TEST" MV_DATE="$D" python code/infer_mv_final.py

echo "=== 2단계: 검증기 재선별 (약 30분) ==="
VER_TARGET=lb VER_TAG=final \
  VER_MV="$OUT/mvresult_BASE_P0_mv8_t0.7_lb_$D.csv" \
  VER_RAW="$OUT/_raw_only_BASE_P0_mv8_t0.7_lb_$D.csv" \
  VER_Q="$TEST" python code/apply_verifier.py

echo "=== 최종 제출본 생성 ==="
python - "$TEST" "$OUT" << 'PY'
import pandas as pd, sys, glob
test, out = sys.argv[1], sys.argv[2]
t = pd.read_csv(test)
a = pd.read_csv(sorted(glob.glob(f"{out}/submission_VERIFIER*.csv"))[-1])[["id","answer"]]
r = t[["id"]].merge(a, on="id", how="left")
assert len(r) == len(t) and r.answer.isna().sum() == 0, "행/결측 불일치"
r["answer"] = r["answer"].astype("int64")
r.to_csv(f"{out}/submission_final.csv", index=False)
print(f"완료: {out}/submission_final.csv  ({len(r)}행)")
PY
