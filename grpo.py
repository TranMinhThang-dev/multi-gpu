import io
import os
import re
from datetime import datetime
import datasets
from math_verify import parse, verify
import json
from tqdm import tqdm
import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import wandb

##############################################################
#                     Global constants                       #
##############################################################
MAX_SEQ_LENGTH = 1024
LOCAL_RANK = os.environ.get("LOCAL_RANK", 0)
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
PER_DEVICE_BATCH_SIZE = 2 // WORLD_SIZE
GRAD_ACCUM = 1

##############################################################
#                       Helper func                          #
##############################################################
def print_batch_configuration(per_device_batch_size, grad_accum, world_size):
    effective_batch = per_device_batch_size * grad_accum * world_size
    print("📊 Batch configuration:")
    print(f"   - World size: {world_size}")
    print(f"   - Per device batch size: {per_device_batch_size}")
    print(f"   - Gradient accumulation: {grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")

print(f"🚀 Initializing training on GPU {LOCAL_RANK}")
print_batch_configuration(PER_DEVICE_BATCH_SIZE, GRAD_ACCUM, WORLD_SIZE)

##############################################################
#                       Prompts                              #
##############################################################
instruction = """
You are a **math, physic, chemiscal Problem-Solving Expert for K12 student**. Your task is to analyze problems and create **concise, detailed solution steps**.
You will serve as a supplementary tool for an LLM, so your output needs to be as **streamlined as possible** for the LLM to understand. No need for lengthy explanations for human readers.

### Your Process:
1. **Analysis & Reasoning (Short)**: Determine problem type, analyze, and identify knowledge needed.
2. **Solution Presentation**: Show steps concisely, end with final answer inside \\boxed{} (math) or after ##### (others).
3. **Principles**: Remove redundant words, write in Vietnamese, full answer, ignore A/B/C labels for choices.
INPUT:
"""

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively."""

##############################################################
#                       Load model                           #
##############################################################
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    torch_dtype = torch.float32,
    gpu_memory_utilization = 0.8,
    device_map = f"cuda:{LOCAL_RANK}",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

##############################################################
#                    Prepare dataset                         #
##############################################################
def convert_to_conversation(sample, subject='math'):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction + '\n' + sample['question']},
        {"role": "assistant", "content": "<think>"}
    ]
    return {
        "prompt": conversation,
        "answer": sample["answer"],
        "question": sample["question"],
        "subject": subject,
        "solution": f"<answer> {sample['answer']} </answer>",
    }

print("📚 Loading and preparing dataset...")

dataset_dict = load_dataset(
    'json',
    data_files={'math': "../data_thinking/correct_data.json"},
)
math_dataset = dataset_dict['math'].map(lambda ex: convert_to_conversation(ex, 'math'))

# 🩹 Remove image-related columns to avoid "images=" error
for col in ["image", "images", "image_path"]:
    if col in math_dataset.column_names:
        math_dataset = math_dataset.remove_columns(col)

def func(example):
    return {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
    }

converted_dataset = math_dataset.map(func)
print("✅ Dataset successfully converted to chat prompt format!")

##############################################################
#                   Reward functions                         #
##############################################################
def language_reward(completions, **kwargs):
    def have_chinese_word(sentence):
        for c in sentence:
            if 0x4E00 <= ord(c) <= 0x9FFF:
                return -1
        return 1
    return [have_chinese_word(c) for c in completions]

def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completions_with_think = ["<think>" + c for c in completions]
    return [1.0 if re.fullmatch(pattern, text, re.DOTALL) else 0.0 for text in completions_with_think]

def accuracy_reward(completions, **kwargs):
    solutions = kwargs.get("solution", [""] * len(completions))
    subjects = kwargs.get("subject", ['math'] * len(completions))
    rewards = []

    for completion, sol, subj in zip(completions, solutions, subjects):
        if subj == 'math':
            pred = parse(completion.replace("</answer>", ""))
            ans = parse(sol.replace("</answer>", ""))
            rewards.append(int(verify(ans, pred)))
        else:
            pred = completion.split("#####")[-1].strip() if "#####" in completion else None
            ans = sol.split("#####")[-1].strip() if "#####" in sol else None
            rewards.append(int(pred == ans if pred and ans else 0))
    return rewards

##############################################################
#                       Train setup                          #
##############################################################
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.002,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_generations = 2,
    max_prompt_length = 4096,
    max_completion_length = 4096,
    num_train_epochs = 1,
    save_steps = 5,
    max_grad_norm = 0.1,
    ddp_find_unused_parameters = False,
    report_to = "wandb",
    output_dir = "outputs",
    min_p = 0.1,
    temperature = 0.6,
    loss_type = "dr_grpo",
)

run = wandb.init(
    project="huggingface",
    name="AI_tutor_GRPO_fixed",
    config=training_args,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,  # ✅ text-only tokenizer, no 'images' arg
    reward_funcs=[
        format_reward,
        accuracy_reward,
        language_reward
    ],
    train_dataset=converted_dataset,
)

trainer.train()
trainer.save_model(training_args.output_dir)
print("🎉 Training complete! Model saved to:", training_args.output_dir)
