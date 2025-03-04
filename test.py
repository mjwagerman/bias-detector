from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

origins = [
    "http://localhost:3000",  # Your Next.js frontend
    # Add other origins if you have multiple frontends
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")

@app.post("/predict")
async def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(dim=-1)[0].tolist()
    return {"bias_scores": probs}  # Returns probabilities for Left, Center, Right

