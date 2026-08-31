# 제5회 대학 연합 딥러닝 챌린지 2026 — 재현 패키지

## 최종 방법

순정 Qwen2.5-3B-Instruct(학습 없음) + Majority Voting 8 + 검증 어댑터 재선별

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 1 | `infer_mv_final.py` | 순정 모델로 문제당 8샘플 생성, format_ok 샘플만 다수결 |
| 2 | `apply_verifier.py` | 4~5표 구간만 검증 어댑터로 재선별. 나머지 구간은 1단계 결과 유지 |

리더보드 0.78339 → **0.78700**

## 실행

```bash
pip install "vllm==0.26.0" "transformers==5.15.0" "trl==1.9.2" \
            "peft==0.20.0" "bitsandbytes==0.50.0" "fsspec==2026.6.0" \
            accelerate datasets huggingface_hub pandas --ignore-installed pyjwt

MV_TARGET=lb MV_INPUT=<test.csv> MV_DATE=0831 python infer_mv_final.py

VER_TARGET=lb VER_TAG=final \
  VER_MV=outputs/mvresult_BASE_P0_mv8_t0.7_lb_0831.csv \
  VER_RAW=outputs/_raw_only_BASE_P0_mv8_t0.7_lb_0831.csv \
  VER_Q=<test.csv> python apply_verifier.py
```

## 추론 설정

vLLM / temperature=0.7 / top_p=0.95 / n=8 / max_tokens=2048 /
max_model_len=3072 / seed=42 / gpu_memory_utilization=0.85

집계: format_ok 샘플만 투표, 합의도 분모는 8 고정
검증기 적용: 4~5표 구간, P(Correct) **합산** 가중투표, margin=0

## 검증 어댑터

- 베이스: Qwen/Qwen2.5-3B-Instruct (규칙 1 준수, 병합 없음)
- LoRA r=16, alpha=32, dropout=0.05, target=q/k/v/o/gate/up/down
- bf16, lr=1e-4, 1 epoch, 실효배치 16, seed=42
- 학습 데이터: `data/verifier_train.jsonl` (9,786샘플, Correct/Incorrect 균형)
  - 대회 train 데이터 3,000문제 표본의 MV8 결과를 gold와 대조해 자동 라벨링
  - **오답 샘플도 학습에 포함** (판별 경계는 오답이 있어야 학습됨)
  - 평가용 홀드아웃 600문제(`data/holdout_ids.json`)는 학습에서 제외
- 가중치: https://huggingface.co/kuenhwan/azu-verifier-r16 (public)

## 사용 데이터

대회 제공 `deep_chal_math_train.csv`만 사용. **외부 데이터셋 미사용.**
리더보드/테스트 문제는 추론 입력으로만 사용했으며 학습에 일절 사용하지 않았습니다.

## 규칙 준수

- 베이스 모델 Qwen/Qwen2.5-3B-Instruct만 사용, 타 모델 가중치 로드·병합 없음
- 검증 어댑터는 동일 베이스에서 자체 학습한 것으로, 외부 모델 앙상블에 해당하지 않음 (규칙 5의 Best-of-N)
- 추론 전 과정 로컬 실행, 외부 API 호출 없음
- 제출 컬럼: 소문자 `id`, `answer`

## 환경

`requirements.txt`, `env.txt`, `gpu.txt` 참조
Docker: `vastai/pytorch_cuda-13.0.3-auto` / vast.ai RTX 3090 24GB

## 검증 어댑터 처음부터 재현하기

```bash
# 1) train 3,000문제로 MV8 생성 (약 70분)
MV_TARGET=train MV_SUBSAMPLE=3000 MV_DATE=0824 python code/infer_mv_final.py

# 2) gold와 대조해 Correct/Incorrect 라벨링 + 홀드아웃 600 분리
python code/make_verifier_data.py

# 3) LoRA 학습 (약 90분)
TRAIN_JSONL=outputs/verifier_train.jsonl RUN=verifier_r16 \
  MAXLEN=1536 python code/train_lora.py
```
