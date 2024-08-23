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

# Load environment variables
load_dotenv()

# Set up AWS credentials from environment variables
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

# Set up Weights & Biases API key
wandb.login(key=os.getenv('WANDB_API_KEY'))

class StreamingYouTubeTitleDataset(Dataset):
    # ... (rest of the class implementation remains the same)

def get_datasets(tokenizer, train_subset_size=None, eval_subset_size=None):
    # ... (function implementation remains the same)

def main(train_subset_size=None, eval_subset_size=None):
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"  # Update this to your actual model path
    
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
        output_dir="s3://your-output-bucket/llama-finetuned-youtube-titles",
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
    trainer.save_model("s3://your-output-bucket/final-model")

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
