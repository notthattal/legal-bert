import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./data/fine_tuned_legalbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict_bill_category(text, label_mapping):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = str(torch.argmax(outputs.logits, dim=1).item())

    return label_mapping[pred]

if __name__ == "__main__":
    with open("./data/label_mapping.json") as f:
        label_mapping = json.load(f)

    result = predict_bill_category('To add Ireland to the E3 nonimmigrant visa program', label_mapping)
    print(f"Bill Category: {result}")