import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# absolute path — works regardless of where streamlit is launched from
MODEL_PATH = r"C:\Users\yoges\OneDrive\Desktop\smai\smai A3\telugu-sentiment\models\telugu-sentiment-final"
print(f"Loading model from: {MODEL_PATH}")
assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"

ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

LABEL_COLORS = {
    "Positive": "🟢 Positive",
    "Negative": "🔴 Negative",
    "Neutral":  "🟡 Neutral"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load once at import time
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def predict_single(text: str):
    """Returns (label, confidence_dict)"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    label = ID2LABEL[pred_id]

    confidence = {
        ID2LABEL[i]: float(probs[i]) for i in range(3)
    }
    return label, confidence


def predict_batch(texts: list):
    """Returns list of (label, confidence_dict)"""
    results = []
    for text in texts:
        label, conf = predict_single(str(text))
        results.append((label, conf))
    return results