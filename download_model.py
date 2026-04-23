from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

REPO     = "YogeshNarasimha/telugu-sentiment-xlm-roberta"
SAVE_DIR = "./notebooks/models/telugu-sentiment-final"

os.makedirs(SAVE_DIR, exist_ok=True)
print("Downloading model (~1.5GB)...")
tok = AutoTokenizer.from_pretrained(REPO)
mdl = AutoModelForSequenceClassification.from_pretrained(REPO)
tok.save_pretrained(SAVE_DIR)
mdl.save_pretrained(SAVE_DIR)
print(f"Done! Model saved to {SAVE_DIR}")