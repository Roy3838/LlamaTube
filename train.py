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
import boto3
from smart_open import smart_open
import json
import random
import wandb
import os
from dotenv import load_dotenv
import logging


logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up AWS credentials from environment variables
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

# Set up Weights & Biases API key
wandb.login(key=os.getenv('WANDB_API_KEY'))

class StreamingYouTubeTitleDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, subset_size=None):
        logger.debug(f"Initializing StreamingYouTubeTitleDataset with file_path={file_path}, max_length={max_length}, subset_size={subset_size}")
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bins = [-np.inf, 1, 2, 3, 4, 5, 6, np.inf]
        self.labels = ["Worst", "Bad", "Average", "Good", "Very Good", "Excellent", "Perfect"]
        self.subset_size = subset_size
        logger.debug("About to load data")
        self.data = self.load_data()
        logger.debug(f"Data loaded, length: {len(self.data)}")

    def load_data(self):
        logger.debug(f"Starting to load data from {self.file_path}")
        data = []
        try:
            with smart_open(self.file_path, 'r') as file:
                for i, line in enumerate(file):
                    if i % 1000 == 0:
                        logger.debug(f"Processed {i} lines")
                    item = json.loads(line)
                    processed_item = self.process_item(item)
                    data.append(processed_item)
                    
                    if self.subset_size and len(data) >= self.subset_size:
                        logger.debug(f"Reached subset size of {self.subset_size}")
                        break
            
            if self.subset_size:
                random.shuffle(data)
                data = data[:self.subset_size]
            
            logger.debug(f"Finished loading data, total items: {len(data)}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_label(self, log_view_count):
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= log_view_count < self.bins[i + 1]:
                return self.labels[i]
        return self.labels[-1]

    def generate_prompt(self, item):
        log_view_count = np.log10(item['view_count'])
        label = self.get_label(log_view_count)
        return f"""Classify the title into one of the following categories: Worst, Bad, Average, Good, Very Good, Excellent, Perfect.
title: {item["title"]}
label: {label}""".strip()

    def process_item(self, item):
        prompt = self.generate_prompt(item)
        inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": inputs.input_ids.squeeze()
        }

def get_datasets(tokenizer, train_subset_size=None, eval_subset_size=None):
    logger.debug(f"Getting datasets with train_subset_size={train_subset_size}, eval_subset_size={eval_subset_size}")
    try:
        train_dataset = StreamingYouTubeTitleDataset('s3://llama-finetuning-data-jay/train_data.jsonl', tokenizer, subset_size=train_subset_size)
        logger.debug(f"Train dataset created, length: {len(train_dataset)}")
    except Exception as e:
        logger.error(f"Error creating train dataset: {str(e)}")
        raise

    try:
        eval_dataset = StreamingYouTubeTitleDataset('s3://llama-finetuning-data-jay/test_data.jsonl', tokenizer, subset_size=eval_subset_size)
        logger.debug(f"Eval dataset created, length: {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"Error creating eval dataset: {str(e)}")
        raise

    return train_dataset, eval_dataset

def main(train_subset_size=None, eval_subset_size=None):
    base_model_name = "/LLAMA_W/1/" # Update this to your actual model path
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
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

    train_dataset, eval_dataset = get_datasets(tokenizer, train_subset_size, eval_subset_size)

    print(f"Training on {len(train_dataset)} samples, evaluating on {len(eval_dataset)} samples")

    training_args = TrainingArguments(
        output_dir="finalmodel",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the model
    trainer.save_model("finalmodel")

    # Example inference
    from transformers import pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    messages = [{"role": "user", "content": "Classify the title into one of the following categories: Worst, Bad, Average, Good, Very Good, Excellent, Perfect.\ntitle: How to make the perfect cup of coffee\nlabel:"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=20, do_sample=True)
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    # Example: Train on 10,000 samples and evaluate on 1,000
    main(train_subset_size=10000, eval_subset_size=1000)
