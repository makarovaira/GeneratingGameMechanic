import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os

dataset_id = "burkelibbey/colors"
base_model_id = "PY007/TinyLlama-1.1B-Chat-v0.3"
model_id_colorist_lora = "mychen76/tinyllama-colorist-lora"

def formatted_train(question, answer) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"


def prepare_train_data(data_id):
    data = load_dataset(data_id, split="train")
    data_df = data.to_pandas()
    data_df["text"] = data_df[["description", "color"]].apply(
        lambda x: "<|im_start|>user\n" + x["description"] + " <|im_end|>\n<|im_start|>assistant\n" + x[
            "color"] + "<|im_end|>\n", axis=1)
    data = Dataset.from_pandas(data_df)
    return data


def get_model_and_tokenizer(mode_id):
    tokenizer = AutoTokenizer.from_pretrained(mode_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        mode_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def finetune_tinyllama(data_id, base_model_id, model_id_colorist_lora):
    data = prepare_train_data(data_id)
    model, tokenizer = get_model_and_tokenizer(base_model_id)

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    training_arguments = TrainingArguments(
        output_dir=model_id_colorist_lora,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        max_steps=200,
        fp16=True,
        push_to_hub=True
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024
    )
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    finetune_tinyllama(dataset_id, base_model_id, model_id_colorist_lora)