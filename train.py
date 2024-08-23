import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import boto3
from smart_open import smart_open


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class StreamingYouTubeTitleDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bins = [-np.inf, 1, 2, 3, 4, 5, 6, np.inf]
        self.labels = ["Worst", "Bad", "Average", "Good", "Very Good", "Excellent", "Perfect"]

    def __iter__(self):
        with smart_open(self.file_path, 'r') as file:
            for line in file:
                item = json.loads(line)
                yield self.process_item(item)

    def get_label(self, log_view_count):
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= log_view_count < self.bins[i + 1]:
                return self.labels[i]
        return self.labels[-1]

    def generate_prompt(self, item):
        log_view_count = np.log10(item['view_count'])
        label = self.get_label(log_view_count)
        return f"""
        Classify the title into one of the following categories: Worst, Bad, Average, Good, Very Good, Excellent, Perfect.
title: {item["title"]}
label: {label}""".strip()

    def process_item(self, item):
        prompt = self.generate_prompt(item)
        log_view_count = np.log10(item['view_count'])
        label = self.get_label(log_view_count)
        label_idx = self.labels.index(label)
        inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": torch.tensor(label_idx, dtype=torch.long)
        }

def get_datasets(tokenizer):
    s3 = boto3.client('s3')
    train_dataset = StreamingYouTubeTitleDataset('s3://your-bucket/train_data.jsonl', tokenizer)
    eval_dataset = StreamingYouTubeTitleDataset('s3://your-bucket/test_data.jsonl', tokenizer)
    return train_dataset, eval_dataset

def main():
    base_model_name = "s3://your-model-bucket/llama-3.1/8b"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype="float16",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = model.to(device)

    train_dataset, eval_dataset = get_datasets(tokenizer)

    training_args = TrainingArguments(
        output_dir="s3://your-output-bucket/llama-3.1-finetuned-youtube-titles",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",  # Using Weights & Biases for experiment tracking
        do_train=True,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("s3://your-output-bucket/final-model")

if __name__ == "__main__":
    main()
