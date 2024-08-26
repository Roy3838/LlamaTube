import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import json
import random
import wandb
import os
from dotenv import load_dotenv
import logging
import pandas as pd
from smart_open import smart_open
from trl import SFTTrainer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up AWS credentials from environment variables
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

# Set up Weights & Biases API key
wandb.login(key=os.getenv('WANDB_API_KEY'))

def load_dataset(file_path, subset_size=None):
    logger.debug(f"Loading dataset from {file_path}")
    titles = []
    view_counts = []
    
    with smart_open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if subset_size and i >= subset_size:
                break
            data = json.loads(line.strip())
            titles.append(data["title"])
            view_counts.append(data["view_count"])
            
            if i % 1000 == 0:
                logger.debug(f"Processed {i} lines")

    # Create a dataset from the dictionary
    dataset_dict = {"title": titles, "view_count": view_counts}
    dataset = Dataset.from_dict(dataset_dict)
    
    logger.debug(f"Dataset loaded, total items: {len(dataset)}")
    return dataset


# Define the prompt format
prediction_prompt = """Predict the order of magnitude of the following video based on the video title:

### Title:
{}

### Views:
{}"""

def formatting_prompts_func(examples, tokenizer):
    titles = examples["title"]
    view_counts = examples["view_count"]
    texts = []
    for title, view_count in zip(titles, view_counts):
        log_views = int(np.log10(view_count))
        text = prediction_prompt.format(title, log_views) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

def get_dataset(tokenizer, file_path, subset_size=None):
    logger.debug(f"Getting dataset with subset_size={subset_size}")
    
    dataset = load_dataset(file_path, subset_size=subset_size)
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
        remove_columns=["title", "view_count"]
    )
    
    logger.debug(f"Dataset processed, total items: {len(dataset)}")
    return dataset

def main(train_subset_size=None):
    base_model_name = "/LLAMA_W/1/" # Update this to your actual model path
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    train_dataset = get_dataset(
        tokenizer,
        's3://llama-finetuning-data-jay/train_data.jsonl',
        train_subset_size
    )

    print(f"Training on {len(train_dataset)} samples")

    training_args = TrainingArguments(
        output_dir="finalmodel",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=512,  # Adjust this value as needed
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer_stats = trainer.train()

    # Save the model
    trainer.save_model("finalmodel")

    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    # Example inference
    pipe = trainer.model.pipeline(
        "text-generation",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    messages = [{"role": "user", "content": "Predict the order of magnitude of the following video based on the video title:\n\n### Title:\nHow to make the perfect cup of coffee\n\n### Views:"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=20, do_sample=True)
    print(outputs[0]["generated_text"])



if __name__ == "__main__":
    main(train_subset_size=10000)
