from datasets import load_dataset
from transformers import AutoTokenizer

print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

print("🧠 First sample from training set:")
print(dataset['train'][0])

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(example):
    return tokenizer(example["text"])

print("⚙️ Tokenizing...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("✅ Tokenization complete.")
print(tokenized_dataset)
