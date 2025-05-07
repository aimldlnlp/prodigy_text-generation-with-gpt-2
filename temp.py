from datasets import load_from_disk

tokenized_datasets = load_from_disk("data/tokenized")

print(tokenized_datasets["train"].features)  # Add this line temporarily