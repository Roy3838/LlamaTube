import torch
from datasets import Dataset
import json
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length, subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = self.load_dataset(file_path, subset_size)

    def load_dataset(self, file_path, subset_size=None):
        titles = []
        view_counts = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if subset_size and i >= subset_size:
                    break
                data = json.loads(line.strip())
                titles.append(data["title"])
                view_counts.append(data["view_count"])
                
                if i % 1000 == 0:
                    print(f"Processed {i} lines")
        
        dataset_dict = {"title": titles, "view_count": view_counts}
        dataset = Dataset.from_dict(dataset_dict)
        print(f"Dataset loaded, total items: {len(dataset)}")
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = self.generate_prompt(item['title'], item['view_count'])
        encoded = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")
        return {
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"]),
            "labels": torch.tensor(encoded["input_ids"])
        }

    @staticmethod
    def generate_prompt(title, view_count):
        log_view_count = np.log10(view_count)
        bins = [-np.inf, 1, 2, 3, 4, 5, 6, np.inf]
        labels = ["Worst", "Bad", "Average", "Good", "Great", "Excellent", "Perfect"]
        label = labels[np.digitize(log_view_count, bins) - 1]
        return f"""Classify the title into one of the following categories: Worst, Bad, Average, Good, Great, Excellent, Perfect. title: {title} label: {label}"""
