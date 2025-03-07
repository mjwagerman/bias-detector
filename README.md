---
tags:
  - bias-detection
  - nlp
  - peft
  - lora
  - fine-tuning
license: mit
datasets:
  - ...
model-index:
  - name: Bias Detector
    results:
      - task:
          type: text-classification
        dataset:
          name: ...
          type: ...
        metrics:
          - type: accuracy
            value: ...
---

# Bias Detector

This model is fine-tuned using **PEFT LoRA** on existing **Hugging Face models** to classify and evaluate the bias in news sources. 

## Model Details
- **Architecture:** Transformer-based (e.g., BERT, RoBERTa)
- **Fine-tuning Method:** Parameter Efficient Fine-Tuning (LoRA)
- **Use Case:** Bias classification, text summarization, sentiment analysis
- **Dataset:** [...](https://huggingface.co/datasets/your-dataset)
- **Training Framework:** PyTorch + Transformers

## Usage
To use this model, install the necessary libraries:
```bash
pip install transformers torch
```
Then load the model with:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "mjwagerman/bias-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "This is an example news headline."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```
