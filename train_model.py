import os
print("📁 Current working directory:", os.getcwd())

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

def tokenize_function(example):
    return tokenizer(example["text"])

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

block_size = 128
def group_texts(examples):
    joined = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(joined["input_ids"])
    total_len = (total_len // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_len, block_size)]
        for k, t in joined.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized.map(group_texts, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"].select(range(200)),  
    eval_dataset=lm_dataset["validation"].select(range(50)),  
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
print("Saving model to nextword-gpt2 folder")
trainer.save_model("nextword-gpt2")
