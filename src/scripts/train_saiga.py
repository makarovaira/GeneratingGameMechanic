import subprocess
import time

import os
import torch
import transformers
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.scripts.dataset import generate_dataset, prepare_data

TOKEN=os.environ['TOKEN']

training_data = generate_dataset("/Users/pancakeswya/sunrise-infinite/test.json")

training_file_name = "training_data.jsonl"

prepare_data(training_data, training_file_name)

MODEL_NAME = "IlyaGusev/saiga2_7b_lora"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subprocess.run(["huggingface-cli", "login", "--token", TOKEN])

st_time = time.time()

config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit = True,
    torch_dtype=torch.float16,
    device_map="auto",
#     quantization_config=bnb_config
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16,
#      torch_dtype=torch.bfloat16,
    is_trainable = True,
#     device_map="auto"
).to(device)


model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)
print(f'Прошло времени {time.time() - st_time}')

data = load_dataset(
    "jsonl",
    data_files={
                'train' : f'./{training_file_name}' ,
    }
)

CUTOFF_LEN = 3584

def generate_prompt(data_point):
    promt = f"""<s>system
{data_point['system']}</s><s>user
{data_point['user']}</s><s>bot
{data_point['assistant']}</s>"""
    #     print(promt)
    return promt


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    #     print(tokenized_full_prompt)
    return tokenized_full_prompt

train_data = (
    data["train"].map(generate_and_tokenize_prompt)
)

BATCH_SIZE = 4
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 100
OUTPUT_DIR = "./tmp"

training_arguments = transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#             warmup_steps=200,
            max_steps=TRAIN_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=10,
            save_steps=10,
            output_dir=OUTPUT_DIR,
            save_total_limit=10,
            load_best_model_at_end=True,
            report_to=None,
            overwrite_output_dir=True, # Overwrite the content of the output dir
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=training_arguments,
    data_collator=data_collator
)

model = torch.compile(model)
trainer.train()
model.save_pretrained(OUTPUT_DIR)