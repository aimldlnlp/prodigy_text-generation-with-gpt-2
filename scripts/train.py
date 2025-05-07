import yaml
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk
import torch

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_dir"])
    tokenized_datasets = load_from_disk("data/tokenized")

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        # learning_rate=config["learning_rate"],
        learning_rate=float(config["learning_rate"]),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        # fp16=False,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_datasets["train"],
    #     tokenizer=tokenizer,
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # <--- Add this line
        tokenizer=tokenizer,
    )


    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()
