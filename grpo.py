import io
import os
import re
from datetime import datetime
import datasets
from math_verify import parse, verify
import json
from tqdm import tqdm
import torch
from PIL import Image
from unsloth import FastVisionModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, concatenate_datasets, Image as ImageFeature

MAX_SEQ_LENGTH = 1024 # Must be this long for VLMs
LORA_RANK = 8 # Larger rank = smarter, but slower
LOCAL_RANK = os.environ.get("LOCAL_RANK", 0)
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
PER_DEVICE_BATCH_SIZE = 2//WORLD_SIZE
GRAD_ACCUM = 1

#############################################################
#                       Helper func                         #
#############################################################
def print_batch_configuration(per_device_batch_size: int, grad_accum: int, world_size: int) -> None:
    effective_batch = per_device_batch_size * grad_accum * world_size
    print("📊 Batch configuration:")
    print(f"   - World size: {world_size}")
    print(f"   - Per device batch size: {per_device_batch_size}")
    print(f"   - Gradient accumulation: {grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")

##############################################################
#                        Initialization                      #
##############################################################

print(f"🚀 Initializing training on GPU {LOCAL_RANK}")
print_batch_configuration(PER_DEVICE_BATCH_SIZE, GRAD_ACCUM, WORLD_SIZE)

global table_data
table_data = []

instruction = """
You are a **math, physic, chemiscal Problem-Solving Expert for K12 student**. Your task is to analyze problems and create **concise, detailed solution steps**.
You will serve as a supplementary tool for an LLM, so your output needs to be as **streamlined as possible** for the LLM to understand. No need for lengthy explanations for human readers.

### Your Process:

1.  **Analysis & Reasoning (Just short enough):**
    * Determine the question type of problem (need to solve the problem or prove the statements). This phase is crucial for understanding the problem.
    * Thoroughly analyze the problem statement and input data (including images, tables, if any). Deeply focus on images, tables, figures,...
    * Determine the **difficulty level** (Basic, Intermediate, Advanced) and the necessary knowledge suitable for a **12th** student.
    * Check the logic and accuracy of the solution.
    * Try to solve first by yourself, then generate the solution steps when solution worked.

2.  **Solution Presentation:**

### Response Structure:

**Problem-Solving Steps:**
    * Focus on question type (e.g., solving problem, proving statement).
    * Steps should be **concise**, not too many small steps. Depends on problem difficulty.
    * List the solution steps and the answer for each step.
    * At the end, provide the **final answer**, highlighted and put your final answer within \\boxed{{}} if it is math question else put your final answer after ##### if it is other subject.
    * Try to use short keyword at each step for shorter response.
    * Highly prefer knowledge K12 textbook. Each step explain a little of what knowledge use

### Limitations & Principles:

* **Remove all superfluous words from the response; keep only key terms as your output is for an LLM.**
* Response in vietnamese
* Exclude steps for reading the problem, concluding the solution, or selecting from multiple-choice options.
* Always provide a complete and exhaustive answer.
* IMPORTANT: **If the question is a multiple-choice question, the final answer just include the full content corresponding to the chosen option within \\boxed{{}} if it is math question else put your final answer after ##### if it is other subject, not include the letter label (e.g., A, B, C, etc.).**
INPUT:
"""
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
                The assistant first thinks about the reasoning process in the mind and then provides the user
                with the answer. The reasoning process and answer are enclosed within <think> </think> and
                <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
                <answer> answer here </answer>"""

##############################################################
#                       Helper function                      #
##############################################################

def path2image(path):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    image = Image.open(path)
    buf = io.BytesIO()
    image.save(buf, format="PNG", quality=95)
    buf.seek(0)
    image = Image.open(buf)
    return image

##############################################################
#                     Load and config model                  #
##############################################################

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "Qwen/Qwen3-0.6B",
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True, # False for LoRA 16bit
    # fast_inference = True, # Enable vLLM fast inference
    torch_dtype = torch.float32, # Use bfloat16 if possible
    gpu_memory_utilization = 0.8, # Reduce if out of memory
    device_map=f"cuda:{LOCAL_RANK}"
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True,  # False if not finetuning language layers
    finetune_attention_modules = True,  # False if not finetuning attention layers
    finetune_mlp_modules       = True,  # False if not finetuning MLP layers

    r = LORA_RANK,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = LORA_RANK*2,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

##############################################################
#                       Prepare dataset                      #
##############################################################

def convert_to_conversation(sample, subject='math'):
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            
        },
        { "role": "user",
          "content" : [
                {"type" : "image"} ,
                {"type" : "text",  "text"  : instruction + '\n'+ sample['question']},
            ]
        },
        {'role':'assistant', 'content':[{"type": "text", "text": "<think>"}]}
    ]
    return { 
            "prompt" : conversation, 
            # "image": sample["image"], 
            "answer": sample["answer"] , 
            'image_path': [sample['image']],  # Store path instead of loaded image
            'question': sample['question'],
            'subject': subject,
            'solution': f"<answer> {sample['answer']} </answer>",
            }


# dataset = []
print("Starting convert to chat prompt template")

# with open("data2/math_train.json",'r') as f:
#     math_dataset = json.load(f)
#     for example in tqdm(math_dataset[:10], desc="Processing math data"):
#         converted_example = convert_to_conversation(sample=example, subject='math')
#         dataset.append(converted_example)

# with open("data2/physic_train.json",'r') as f:
#     physic_dataset = json.load(f)
#     for example in tqdm(physic_dataset[:10], desc="Processing physics data"):
#         converted_example = convert_to_conversation(sample=example, subject='physic')
#         dataset.append(converted_example)
    
    
# with open("data2/chemistry_train.json",'r') as f:
#     chemistry_dataset = json.load(f)
#     for example in tqdm(chemistry_dataset[:10], desc="Processing chemistry data"):
#         converted_example = convert_to_conversation(sample=example, subject='chemistry')
#         dataset.append(converted_example)


dataset_dict = load_dataset(
    'json', 
    data_files={
        'math': "/kaggle/input/thinking-data2/data_thinking/correct_data.json", 
    },
)

math_dataset = dataset_dict['math'].map(lambda example: convert_to_conversation(example, 'math'), batched=False)

# math_dataset = math_dataset.cast_column("image", ImageFeature())
# chemistry_dataset = chemistry_dataset.cast_column("image", ImageFeature())
# physic_dataset = physic_dataset.cast_column("image", ImageFeature())

print("Number of math sample:", len(math_dataset))
# print("Number of total samples: ", len(dataset))

# converted_dataset = datasets.Dataset.from_list(dataset)

# converted_dataset = concatenate_datasets([math_dataset.select(range(10)), physic_dataset.select(range(10)), chemistry_dataset.select(range(10))])
converted_dataset = concatenate_datasets([math_dataset])

def func(example):
    print(example)
    return {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
    }

converted_dataset = converted_dataset.map(
    lambda example: func(example)
)
print("Convert to chat prompt template succesfully!!!")

##############################################################
#                       Reward functions                     #
##############################################################
def language_reward(completions, **kwargs):
    def have_chinese_word(sentense: str):
        for c in sentense:
            if  0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF or 0xF900 <= ord(c) <= 0xFAFF:
                return -1
        return 1
    
    completion_contents = [completion for completion in completions]
    rewards = []
    for completion_content in completion_contents:
        rewards.append(have_chinese_word(completion_content))
    
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = ["<think>" + completion for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, **kwargs):
    solutions = kwargs.get("solution",["" for _ in range(len(completions))])
    subjects = kwargs.get("subject",['math' for _ in range(len(completions))])
    completion_contents = [completion for completion in completions]
    questions = kwargs.get("question",["" for _ in range(len(completions))])
    images = kwargs.get("image_path",["" for _ in range(len(completions))])
    for i in range(len(questions)):
        table_data.append(
            [wandb.Image(images[i][0]), questions[i], completion_contents[i]]
        )
    logging_table = wandb.Table(columns=["images", "questions", "completions"], data=table_data)
    wandb.log({"logging table": logging_table})
    rewards = []

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_accuracy.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Accuracy reward -------------\n")
            for content, solution in zip(completions, solutions):
                f.write(f"answer: {content}\n")
                f.write(f"solution: {solution}\n")
    # print("completions: ", completion_contents)
    # print("solutions: ", solutions)
    for i in range(len(completion_contents)):
        processed_solution = str(solutions[i]).replace("</answer>","")
        processed_completion_contents = str(completion_contents[i]).replace("</answer>","")
        if subjects[i] == 'math':
            predict = parse(processed_completion_contents)
            answer = parse(processed_solution)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_accuracy.txt"), "a", encoding='utf-8') as f:
                        f.write(f"PARSED ANDWER: {answer}\n")
                        f.write(f"PARSED PREDICT: {predict}\n")
            is_correct = verify(answer, predict)
            rewards.append(int(is_correct))
        else:
            predict = None
            answer = None
            if "#####" in processed_completion_contents:
                pre_index = processed_completion_contents.index("#####")
                predict = processed_completion_contents[pre_index: ].replace("#####","").strip()
            if "#####" in processed_solution:
                ans_index = processed_solution.index("#####")
                answer = processed_solution[ans_index: ].replace("#####","").strip()
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_accuracy.txt"), "a", encoding='utf-8') as f:
                        f.write(f"PARSED ANDWER: {answer}\n")
                        f.write(f"PARSED PREDICT: {predict}\n")
            if answer is not None and predict is not None and answer == predict:
                rewards.append(int(answer == predict))
            else:
                rewards.append(0)

    return rewards

##############################################################
#                           Inference                        #
##############################################################

# from vllm import SamplingParams
# sampling_params = SamplingParams(
#     temperature = 1.0,
#     top_k = 50,
#     max_tokens = 8096,
# )

# outputs = model.fast_generate(
#     {
#         "prompt": converted_dataset[0]["prompt"],
#         "multi_modal_data": {"image": converted_dataset[0]["image"]}
#     },
#     sampling_params,
# )
# print(outputs[0].outputs[0].text)


##############################################################
#                    Training and config                     #
##############################################################

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    # deepspeed="zero3.json", # Use ZeRO-3
    learning_rate = 5e-6,
    # bf16=True,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.002,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    log_completions = False,
    per_device_train_batch_size = PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = 4096,
    max_completion_length = 4096,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps = 60,
    save_steps = 5,
    max_grad_norm = 0.1,
    ddp_find_unused_parameters=False, # enable distributed training
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
    min_p=0.1,
    temperature=0.6,

    # Below enables GSPO:
    # importance_sampling_level = "sequence",
    mask_truncated_completions = False,
    loss_type = "dr_grpo",
)

import wandb
run = wandb.init(
        project="huggingface",
        name="AI_tutor_GRPO",
        config=training_args,
        )

trainer = GRPOTrainer(
    model = model,
    args = training_args,
    # Pass the processor to handle multimodal inputs
    processing_class = tokenizer,
    reward_funcs = [
        format_reward,
        accuracy_reward,
        language_reward
    ],
    train_dataset = converted_dataset,
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()
trainer.save_model(training_args.output_dir)