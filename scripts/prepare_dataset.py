import yaml
from transformers import AutoTokenizer
from datasets import load_dataset
import os

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def prepare_tokenizer(model_name, tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad_token to eos_token (end of sequence token)
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer

# def tokenize_function(examples, tokenizer, max_length):
#     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

# def tokenize_function(examples, tokenizer, max_length):
#     output = tokenizer(
#         examples["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=max_length,
#     )
#     # Add labels that are identical to input_ids for causal LM
#     output["labels"] = output["input_ids"].copy()
#     return output

def tokenize_function(examples, tokenizer, max_length):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # Add labels: same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def split_datasets(raw_datasets, test_size=0.1):
    # Split the dataset into train and test sets
    return raw_datasets["train"].train_test_split(test_size=test_size)

def main():
    config = load_config()
    tokenizer = prepare_tokenizer(config["model_name"], config["tokenizer_dir"])
    
    raw_datasets = load_dataset('text', data_files={"train": config["train_file"]})
    split_dataset = split_datasets(raw_datasets, test_size=0.1)  # 90% train, 10% validation
    
    tokenized_datasets = split_dataset.map(lambda x: tokenize_function(x, tokenizer, config["max_length"]), batched=True)
    tokenized_datasets.save_to_disk("data/tokenized")

if __name__ == "__main__":
    main()
