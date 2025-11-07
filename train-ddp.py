#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA fine-tuning for Qwen3-14B with Unsloth + DDP (torchrun)
————————————————————————————————————————————————————————————
Usage (example, 4x GPUs):
  torchrun --nproc_per_node=4 train_ddp.py

Notes:
- Each torchrun process is pinned to a single GPU via LOCAL_RANK.
- We set `ddp_find_unused_parameters=False` which is common for PEFT/LoRA
  with gradient checkpointing to reduce DDP sync overhead/conflicts.
- The script keeps logging mostly on rank 0 to avoid noisy output.
"""

import os
import torch
import logging
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# --------------------------- DDP bootstrap ---------------------------
LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", 0))
RANK        = int(os.environ.get("RANK", 0))
WORLD_SIZE  = int(os.environ.get("WORLD_SIZE", 1))
IS_DIST     = WORLD_SIZE > 1

if IS_DIST:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

# Less noise on non-master ranks
logging.basicConfig(level=logging.INFO if RANK == 0 else logging.WARNING)
log = logging.getLogger("train_ddp")
log.info(f"RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE} device={device}")


# --------------------------- User config -----------------------------
LORA_RANK      = 32
NUM_EPOCHS     = 3
MODEL_SIZE_B   = 14
MAX_LEN        = 4096
MODEL_PATH     = "unsloth/Qwen3-0.6B"
DATA_FILE      = "./science_sft_subset5k.jsonl"
LR             = 2e-4
SAVE_STEPS     = 20000
RESUME_FROM    = None  # e.g., "checkpoints_xxx/checkpoint-XXXX"

# We target an effective global batch of 16 (adjust to your memory).
TARGET_GLOBAL_BATCH = 16
per_device_bs = max(1, TARGET_GLOBAL_BATCH // max(1, WORLD_SIZE))
grad_accum    = 1  # increase this if you want bigger global batch without OOM
log.info(f"per_device_train_batch_size={per_device_bs}, gradient_accumulation_steps={grad_accum}")

# ---------------------- 1) Load model/tokenizer ----------------------
# Data-parallel load: each process places the model on its own GPU.
device_map = {"": f"cuda:{LOCAL_RANK}"} if torch.cuda.is_available() else None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name        = MODEL_PATH,
    max_seq_length    = MAX_LEN,
    load_in_4bit      = True,
    load_in_8bit      = False,
    full_finetuning   = False,
    device_map        = device_map,
)

# Wrap with LoRA (PEFT). Keep your original target modules.
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    target_modules             = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha                 = LORA_RANK,
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)

# -------------------- 2) Read data & build chats ---------------------
# Expect a JSONL with fields: "instruction" and "output".
ds = load_dataset("json", data_files=DATA_FILE, split="train")

def to_conversations(batch):
    """
    Build chat-format: [[{role:user, content:...}, {role:assistant, content:...}], ...]
    """
    convs = []
    for q, a in zip(batch["instruction"], batch["output"]):
        convs.append([{"role": "user", "content": q},
                      {"role": "assistant", "content": a}])
    return {"conversations": convs}

ds = ds.map(to_conversations, batched=True, desc="Build conversations")

# --------------- 3) Apply chat template & truncate -------------------
def apply_template_and_trunc(ex):
    """
    Apply tokenizer chat template, then truncate to MAX_LEN and store as plain text.
    Keeping text-only aligns with TRL's dataset_text_field usage.
    """
    text = tokenizer.apply_chat_template(ex["conversations"], tokenize=False)
    ids = tokenizer(
        text, truncation=True, max_length=MAX_LEN, add_special_tokens=False
    )["input_ids"]
    ex["text"] = tokenizer.decode(ids)
    return ex

# num_proc can be tuned to your CPU; 8 is a safe default on many servers.
ds = ds.map(apply_template_and_trunc, num_proc=8, desc="Template + truncate")
dataset = ds.remove_columns([c for c in ds.column_names if c != "text"]).shuffle(seed=3407)

# ---------------- 4) SFT config (DDP-friendly settings) --------------
out_dir = f"./checkpoints_{MODEL_SIZE_B}B_rank_{LORA_RANK}"

cfg = SFTConfig(
    dataset_text_field           = "text",
    per_device_train_batch_size  = per_device_bs,
    gradient_accumulation_steps  = grad_accum,
    num_train_epochs             = NUM_EPOCHS,
    learning_rate                = LR,
    warmup_ratio                 = 0.03,
    lr_scheduler_type            = "linear",
    max_grad_norm                = 1.0,
    weight_decay                 = 0.001,
    seed                         = 3407,
    output_dir                   = out_dir,
    save_strategy                = "steps",
    save_steps                   = SAVE_STEPS,
    save_total_limit             = 20,
    logging_steps                = 5,
    report_to                    = "none",
    optim                        = "adamw_8bit",
    ddp_find_unused_parameters   = False,  # important for LoRA + checkpointing
)

# -------------------- 5) Optional safety callback --------------------
class GradientGuard(TrainerCallback):
    """Skip a step if grad_norm explodes (quick & dirty guard)."""
    def on_step_end(self, args, state, control, **kwargs):
        gn = kwargs.get("logs", {}).get("grad_norm")
        if gn is None and state.log_history:
            gn = state.log_history[-1].get("grad_norm")
        if gn is not None and gn > 100:
            if RANK == 0:
                print(f"[Guard] grad_norm={gn:.1f} -> skip step {state.global_step}")
            control.should_skip_next_step = True

# --------------------- 6) Trainer & training loop --------------------
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args          = cfg,
    callbacks     = [GradientGuard()],
)

if RANK == 0:
    log.info("Start training on main process...")
train_stats = trainer.train(resume_from_checkpoint=RESUME_FROM)

# --------------------- 7) Save only on rank 0 ------------------------
if RANK == 0:
    save_dir = f"./lora_model_{MODEL_SIZE_B}B_rank_{LORA_RANK}_epoch_{NUM_EPOCHS}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Training complete. Saved to: {save_dir}")

# ----------------------- Graceful DDP cleanup ------------------------
if IS_DIST and torch.distributed.is_initialized():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
