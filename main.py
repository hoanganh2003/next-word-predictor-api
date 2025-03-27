from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="Next Word Predictor", description="Predicts the next word", version="1.0")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class Prompt(BaseModel):
    text: str

@app.post("/predict")
def predict_next_word(prompt: Prompt):
    input_text = prompt.text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=input_ids.shape[1]+1, do_sample=False)

    predicted_token_id = output[0][-1].item()
    predicted_word = tokenizer.decode(predicted_token_id)

    return {
        "input": input_text,
        "next_word": predicted_word.strip()
    }
