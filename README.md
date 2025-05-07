# Text Generation with GPT-2 Fine-Tuning

This project provides a framework for fine-tuning the GPT-2 language model on a custom dataset to perform text generation tasks. It leverages the Hugging Face Transformers library and the Datasets library to train, evaluate, and generate text using a GPT-2 model fine-tuned on your own data.

## Features

- Fine-tune GPT-2 on custom datasets
- Configurable training parameters via YAML config file
- Training and evaluation using Hugging Face Trainer API
- Text generation with sampling techniques (top-k, top-p)
- Support for GPU acceleration if available

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd prodigy_text-generation-with-gpt-2
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Training and model parameters are configured in `config/config.yaml`. Key parameters include:

- `model_name`: Pretrained model name (default: `gpt2`)
- `train_file`: Path to the training dataset file
- `output_dir`: Directory to save the fine-tuned model and tokenizer
- `tokenizer_dir`: Directory of the tokenizer to use
- `max_length`: Maximum sequence length for training
- `train_batch_size`: Batch size for training
- `eval_batch_size`: Batch size for evaluation
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `logging_steps`, `save_steps`, `eval_steps`: Steps intervals for logging, saving, and evaluation

## Usage

### Training the Model

Run the training script to fine-tune GPT-2 on your dataset:

```bash
python scripts/train.py
```

The script loads the configuration from `config/config.yaml`, loads the dataset from the `data/tokenized` directory, and trains the model. The fine-tuned model and tokenizer will be saved to the directory specified by `output_dir`.

### Generating Text

After training, generate text using the fine-tuned model:

```bash
python scripts/generate.py
```

You will be prompted to enter a text prompt. The model will generate and display text based on the prompt.

## Dependencies

- transformers
- datasets
- torch
- tqdm
- pyyaml

All dependencies are listed in `requirements.txt` and can be installed via pip.

## Data

- Training data should be placed in `data/your_dataset.txt`.
- Tokenized datasets are stored in `data/tokenized/`.

## Model and Tokenizer

- The pretrained GPT-2 model is fine-tuned and saved in `models/gpt2-finetuned/`.
- The tokenizer files are located in `models/tokenizer/`.

## License

This project does not include a license. Please add a license file if you intend to distribute or publish this project.
