import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    config = load_config()
    model = AutoModelForCausalLM.from_pretrained(config["output_dir"]).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])

    prompt = input("Enter your prompt: ")
    generated = generate_text(prompt, model, tokenizer)
    print("\nGenerated Text:\n", generated)

if __name__ == "__main__":
    main()
