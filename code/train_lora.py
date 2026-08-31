# -*- coding: utf-8 -*-
"""bf16 LoRA 학습 (양자화 없음). Qwen2.5-3B-Instruct + LoRA 어댑터.

QLoRA(4bit)를 쓰지 않는 이유:
  - QLoRA 방침은 8/2에 Colab T4 16GB를 전제로 정한 것입니다.
    3090 24GB에서는 3B를 bf16으로 풀로드해도(가중치 6GB) 여유가 있습니다.
  - 더 중요한 건 학습/추론 정합성입니다. 추론(infer_mv.py)은 bf16으로 도는데
    학습만 4bit로 하면, 어댑터가 4bit 양자화 오차를 보정하도록 학습된 뒤
    그 오차가 없는 환경에 얹히게 됩니다.

TRL을 쓰지 않는 이유:
  - 라벨 마스킹(어디까지를 학습에서 제외하는가)이 이 스크립트에서 가장
    중요한 부분인데, 라이브러리에 맡기면 눈으로 확인할 수 없습니다.
    아래 build_example()에서 직접 처리하고, 학습 전에 샘플을 출력해 검증합니다.

실행:
  TRAIN_JSONL=/workspace/outputs/train_sft.jsonl \
  RUN=lora_bf16_r16 python train_lora.py

주요 환경변수 (기본값은 아래 CFG 참조):
  TRAIN_JSONL  학습 데이터 (필수)
  RUN          실험 이름 → /workspace/checkpoints/{RUN}/
  LORA_R  LORA_ALPHA  LR  EPOCHS  BATCH  ACCUM  MAXLEN  SEED
"""

import os, sys, json, time, random
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/workspace")
from infer_baseline import MODEL_NAME, SYSTEM_PROMPT

# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
CFG = dict(
    train_jsonl = os.environ["TRAIN_JSONL"],
    run         = os.environ.get("RUN", "lora_bf16_r16"),
    lora_r      = int(os.environ.get("LORA_R", 16)),
    lora_alpha  = int(os.environ.get("LORA_ALPHA", 32)),
    lora_drop   = float(os.environ.get("LORA_DROPOUT", 0.05)),
    lr          = float(os.environ.get("LR", 1e-4)),
    epochs      = float(os.environ.get("EPOCHS", 1)),
    batch       = int(os.environ.get("BATCH", 4)),
    accum       = int(os.environ.get("ACCUM", 4)),
    maxlen      = int(os.environ.get("MAXLEN", 1024)),
    seed        = int(os.environ.get("SEED", 42)),
)
OUT_DIR = f"/workspace/checkpoints/{CFG['run']}"
FINAL   = f"{OUT_DIR}/final"

# 학습률을 낮게(1e-4) 잡은 이유:
#   지금 최고 성적(LB 0.78339)은 Majority Voting이 만든 것이고, MV는 8개 샘플이
#   서로 다른 실수를 해야 작동합니다. 학습이 세게 들어가면 분포가 뾰족해져
#   만장일치가 늘고(발견 6: 0.4633 -> 0.5513) MV의 이득이 사라집니다.
#   성능이 안 나오면 LR을 올리기 전에 먼저 만장일치율부터 확인하세요.

if os.path.exists(FINAL):
    raise SystemExit(f"[중단] 이미 존재합니다: {FINAL}\n  RUN 이름을 바꾸세요.")
if "val_sft" in CFG["train_jsonl"]:
    raise SystemExit("[중단] val 데이터로 학습하면 평가셋이 오염됩니다.")

print("=" * 70)
for k, v in CFG.items():
    print(f"  {k:12s} = {v}")
print(f"  effective batch = {CFG['batch'] * CFG['accum']}")
print("=" * 70)

set_seed(CFG["seed"])
random.seed(CFG["seed"])

# ------------------------------------------------------------
# 데이터
# ------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def build_example(msgs):
    """messages -> {input_ids, labels}. 질문 부분은 labels=-100으로 가려서
    모델이 '질문을 생성하는 법'이 아니라 '답을 생성하는 법'만 배우게 합니다."""
    sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), SYSTEM_PROMPT)
    user   = next(m["content"] for m in msgs if m["role"] == "user")
    asst   = next(m["content"] for m in msgs if m["role"] == "assistant")

    # 추론 때와 글자 단위로 같은 프롬프트여야 합니다 (infer_mv.py와 동일 경로)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    # 어시스턴트 턴 종료 토큰까지 포함해야 모델이 언제 멈출지 배웁니다
    completion = asst + "<|im_end|>\n"

    p = tok(prompt,     add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    return {"input_ids": p + c, "labels": [-100] * len(p) + c, "sys": sysmsg}


rows, n_bad, n_long = [], 0, 0
with open(CFG["train_jsonl"], encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            ex = build_example(json.loads(line)["messages"])
        except (KeyError, StopIteration, json.JSONDecodeError):
            n_bad += 1
            continue
        # 길이 초과는 자르지 말고 버립니다.
        # 자르면 끝의 'Answer: N'이 날아가서, 모델에게 "답을 내지 마라"를
        # 가르치는 꼴이 됩니다. 이건 조용히 성능을 갉아먹습니다.
        if len(ex["input_ids"]) > CFG["maxlen"]:
            n_long += 1
            continue
        rows.append(ex)

if not rows:
    raise SystemExit("학습 샘플이 0개입니다. TRAIN_JSONL 경로/형식을 확인하세요.")

sys_set = {r.pop("sys") for r in rows}
lens = sorted(len(r["input_ids"]) for r in rows)
print(f"\n[데이터] 사용 {len(rows)}개 / 형식오류 {n_bad} / 길이초과 {n_long}")
print(f"  길이 중앙값 {lens[len(lens)//2]}, 95% {lens[int(len(lens)*0.95)]}, 최대 {lens[-1]}")
if len(sys_set) > 1:
    print(f"  [경고] 시스템 프롬프트가 {len(sys_set)}종류 섞여 있습니다")
elif sys_set and next(iter(sys_set)) != SYSTEM_PROMPT:
    print("  [경고] 학습 데이터의 시스템 프롬프트가 추론용과 다릅니다")
    print("         학습/추론 불일치는 성능을 조용히 떨어뜨립니다")

# 마스킹이 의도대로 됐는지 눈으로 확인 (가장 중요한 검증)
s = rows[0]
n_sup = sum(1 for l in s["labels"] if l != -100)
print(f"\n[마스킹 확인] 첫 샘플 총 {len(s['input_ids'])}토큰 중 학습 대상 {n_sup}토큰")
print(f"  학습 대상 시작: {tok.decode([l for l in s['labels'] if l != -100][:40])!r}")
print(f"  학습 대상 끝  : {tok.decode([l for l in s['labels'] if l != -100][-20:])!r}")
print("  -> 앞은 풀이 시작, 끝은 'Answer: N<|im_end|>'이어야 정상입니다\n")

random.shuffle(rows)


def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {
        "input_ids":      torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
        "labels":         torch.tensor([b["labels"]    + [-100] * (m - len(b["labels"]))    for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]),
    }


# ------------------------------------------------------------
# 모델
# ------------------------------------------------------------
print("[모델] bf16 풀로드 (양자화 없음)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.config.use_cache = False          # gradient checkpointing과 충돌 방지
model.enable_input_require_grads()      # PEFT + checkpointing 조합에 필요

model = get_peft_model(model, LoraConfig(
    r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_drop"],
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()
print(f"  GPU 사용: {torch.cuda.memory_allocated()/1e9:.1f}GB / 24GB")

# ------------------------------------------------------------
# 학습
# ------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=CFG["batch"],
        gradient_accumulation_steps=CFG["accum"],
        num_train_epochs=CFG["epochs"],
        learning_rate=CFG["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=25,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,     # 디스크가 금방 찹니다
        seed=CFG["seed"],
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
    ),
    train_dataset=rows,
    data_collator=collate,
)

t0 = time.time()
trainer.train()
model.save_pretrained(FINAL)
tok.save_pretrained(FINAL)

with open(f"{FINAL}/train_config.json", "w") as f:
    json.dump({**CFG, "n_samples": len(rows), "base": MODEL_NAME,
               "quantization": "none (bf16)"}, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 70}")
print(f"완료: {FINAL}   ({(time.time()-t0)/60:.1f}분)")
print(f"\n다음 단계:")
print(f"  1) 백업")
print(f"     hf upload kuenhwan/azu-deeplearning-cot {FINAL} checkpoints/{CFG['run']} --repo-type dataset")
print(f"  2) 평가 (온도 스윕 — 발견 6에서 QLoRA 최적점은 0.7이 아니라 1.0이었음)")
for t in ("0.7", "1.0", "1.2"):
    print(f"     MV_LORA={FINAL} MV_TAG={CFG['run']} MV_TEMP={t} "
          f"MV_LORA_RANK={CFG['lora_r']} MV_TARGET=val MV_DATE=0822 python infer_mv.py")
print(f"  3) 끝나면 인스턴스 stop")
print("=" * 70)
