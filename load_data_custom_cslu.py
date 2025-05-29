from datasets import Dataset, DatasetDict
import os

def load_data_custom_cslu(dataset_path, mode="train"):
    if mode == "train":
        print(f"Loading train dataset from {dataset_path}/train")
        train_dataset = Dataset.load_from_disk(f"{dataset_path}/train")
        print(f"Loading development dataset from {dataset_path}/development")
        dev_dataset = Dataset.load_from_disk(f"{dataset_path}/development")

        custom_dataset = DatasetDict({"train": train_dataset.shuffle(), "development": dev_dataset})
        print("Train and development datasets loaded")
        print(f"Found {len(custom_dataset['train'])} training examples and {len(custom_dataset['development'])} development examples")
    
    elif mode == "test":
        print(f"Loading test dataset from {dataset_path}/test")
        test_dataset = Dataset.load_from_disk(f"{dataset_path}/test")

        custom_dataset = DatasetDict({"test": test_dataset})
        print("Test dataset loaded")
        print(f"Found {len(custom_dataset['test'])} test examples")
    
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    return custom_dataset