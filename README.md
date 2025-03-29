This is a simple NLP-powered API that predicts the next word based on a given prompt using GPT-2. Built using Python, FastAPI, Hugging Face Transformers, Docker, and deployed to Google Cloud Run.

---

Features

- Predicts the most likely next word from a given sentence
- Built with FastAPI and Transformers (GPT-2)
- Deployed serverlessly on Google Cloud Run
- Fully containerized with Docker

---

Target Users

NLP Students - Learn how language models generate predictions 
Developers - Use it as a building block for smart autocomplete 
Educators - Demonstrate language modeling in real time 

---

Dataset

- **Wikitext-2 (raw)** from Hugging Face Datasets  

---

- Expected Usage Volume: ~100–300 requests/day
- User Requirements: Real-time processing with response times under 1 second per request

## Model Training

| Config               | Value             |
|----------------------|-------------------|
| Model                | GPT-2 (base)      |
| Epochs               | 1                 |
| Batch Size           | 8                 |
| Max Seq Length       | 128               |
| Learning Rate        | 2e-5              |
| Library              | Transformers   |


---

[https://nextword-api-446334607314.us-central1.run.app/docs](https://nextword-api-446334607314.us-central1.run.app/docs)

Sample input JSON:
```json
{
  "text": "string"
}

{
  "input": "Machine learning is",
  "next_word": " a"
}
