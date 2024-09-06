import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from llama_recipes.configs import train_config as TRAIN_CONFIG

train_config = TRAIN_CONFIG()
train_config.model_name = "/home/ubuntu/LLAMA_W/1"
train_config.num_epochs = 1
train_config.run_validation = False
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.output_dir = "meta-llama-overnight1.0"
train_config.use_peft = True
train_config.max_seq_length = 2048

from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
)


#|%%--%%| <7KGZbxzkfN|fWrJs07DTq>

import torch
import gc

# Delete existing model if it exists
if 'model' in globals():
    del model

# Clear CUDA cache
torch.cuda.empty_cache()

# Garbage collect
gc.collect()

print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

#|%%--%%| <fWrJs07DTq|45nwxVDDlx>

model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            quantization_config=config,
            use_cache=False,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            torch_dtype=torch.float16,
        )


tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token

#|%%--%%| <45nwxVDDlx|aQTgqdTzH0>

eval_prompt = """
The meaning of life and everything is:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.inference_mode():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))


#|%%--%%| <aQTgqdTzH0|gCDW45FcD4>

from datasets import Dataset
import json
import numpy as np
from torch.utils.data import DataLoader


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

def get_dataloader(tokenizer, file_path, train_config, split="train"):
    dataset = CustomDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_length=train_config.max_seq_length,
        subset_size=train_config.subset_size if hasattr(train_config, 'subset_size') else None
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size_training if split == "train" else train_config.batch_size_eval,
        shuffle=(split == "train"),
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True
    )
    
    return dataloader

#|%%--%%| <gCDW45FcD4|Y24jtE24RY>

train_config.subset_size = 10000
train_config.batch_size_eval = 8
train_dataloader = get_dataloader(tokenizer, "/home/ubuntu/data/data.jsonl", train_config, "train")
eval_dataloader = get_dataloader(tokenizer, "/home/ubuntu/data/data.jsonl", train_config, "val")

#|%%--%%| <Y24jtE24RY|leBNWIZRTG>

from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_recipes.configs import lora_config as LORA_CONFIG

lora_config = LORA_CONFIG()
lora_config.r = 8
lora_config.lora_alpha = 32
lora_dropout: float=0.01

peft_config = LoraConfig(**asdict(lora_config))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

#|%%--%%| <leBNWIZRTG|pzh0vVUS0u>

import torch.optim as optim
from llama_recipes.utils.train_utils import train
from torch.optim.lr_scheduler import StepLR

model.train()

optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

# Start the training process
results = train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    scheduler,
    train_config.gradient_accumulation_steps,
    train_config,
    None,
    None,
    None,
    wandb_run=None,
)

#|%%--%%| <pzh0vVUS0u|OwkPQSwTQH>

model.save_pretrained(train_config.output_dir)



