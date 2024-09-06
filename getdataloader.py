from torch.utils.data import DataLoader

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
