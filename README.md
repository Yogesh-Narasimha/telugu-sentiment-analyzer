# 🎬 Telugu Sentiment Analyzer — T8.3

**SMAI Assignment 3 | IIIT Hyderabad 2025–26 | Team: Tech Titans**

A fine-tuned `xlm-roberta-large` model for Telugu sentiment analysis across 3 classes: **Positive**, **Negative**, and **Neutral**. Includes a Streamlit web app with single review analysis, bulk CSV processing, and Gemini-powered theme extraction.

---

## Results

| Model | Overall | Neutral | Notes |
|-------|---------|---------|-------|
| V1 indic-bert + 61 neutral | 70% | 10% | Baseline |
| V2 indic-bert + 200 neutral | 70% | 10% | More data, same model — no improvement |
| tabularisai multilingual | 70% | 10% | External baseline |
| **V3 xlm-roberta-large ★** | **80%** | **60%** | Best model — TA approved |

**Key finding:** Model size (560M params) matters more than data augmentation for Telugu Neutral detection.

---
## Live Demo
🔗 [HuggingFace Spaces](https://huggingface.co/spaces/YogeshNarasimha/telugu-sentiment-analyzer)

---
## App Features

- **Single Review** — paste any Telugu review, get Positive/Negative/Neutral with confidence bar
- **Bulk CSV** — upload CSV, label all reviews, dashboard with 5 charts
- **Gemini Theme Extraction** — extract top 5 themes from bulk reviews via Gemini API
- **Model Comparison** — switch between V3 (best), multilingual baseline, indic-bert baseline
- **Short review warning** — alerts when review is too short for accurate prediction

---

## Project Structure

```
telugu-sentiment/
├── app/
│   └── app.py                   # Streamlit web app
├── notebooks/
│   ├── train.ipynb              # Training pipeline + ablation study
│   ├── telugu_test_labeled.csv  # 30-review labeled benchmark
│   ├── cache/
│   │   └── neutral_200.json     # 200 Gemini-generated neutral reviews
│   └── report/
│       ├── model_comparison.png # Model comparison chart
│       └── results_analysis.png # Ablation + confusion matrix
├── download_model.py            # Download model from HuggingFace Hub
├── requirements.txt
└── README.md
```

---

## Setup & Run Locally

### Prerequisites
- Python 3.10
- Git
- ~3GB free disk space (for model)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Yogesh-Narasimha/telugu-sentiment-analyzer.git
cd telugu-sentiment-analyzer

# 2. Create virtual environment
py -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the model from HuggingFace Hub (~1.5GB, takes 3-5 mins)
python download_model.py

# 5. Run the app
streamlit run app/app.py
```

App opens at `http://localhost:8501`

---

## Training Pipeline

The notebook `notebooks/train.ipynb` covers:

1. Load `ai4bharat/IndicSentiment` Telugu config
2. Generate 200 neutral reviews via Gemini API (dataset has no Neutral labels)
3. Train V1 — indic-bert baseline
4. Train V2 — indic-bert + 200 balanced neutral + class weights
5. Train V3 — xlm-roberta-large + 200 balanced neutral + class weights
6. Benchmark all 3 models on 30-review labeled test set
7. Hybrid inference — V3 + Gemini fallback for low-confidence predictions
8. Error analysis + ablation study

---

## Dataset

| Split | Source | Size |
|-------|--------|------|
| Train (Pos/Neg) | ai4bharat/IndicSentiment Telugu | 998 reviews |
| Train (Neutral) | Gemini-generated synthetic | 180 reviews |
| Eval | ai4bharat/IndicSentiment validation | 156 reviews |
| Benchmark | Hand-labeled (10 per class) | 30 reviews |

---

## Experiment Journey

```
Problem   → indic-bert (33M) fails on Neutral — F1 = 0.00
Hypothesis 1 → More neutral data will fix it
Result 1  → V2 (200 neutral) → still 0% Neutral ✗
Hypothesis 2 → Bigger model will fix it  
Result 2  → V3 xlm-roberta-large (560M) → 60% Neutral ✓
Conclusion → Model capacity matters more than data quantity
             for low-resource Telugu Neutral detection
```

---

## Known Limitations

- **Neutral recall** — 60% accuracy, 4 high-confidence wrong predictions remain
- **Code-mixed loanwords** — Telugu-English words like *వీక్గా*, *వేస్ట్* sometimes misclassified
- **Short reviews** — fewer than 5 words gives unreliable predictions
- **Domain shift** — trained on product reviews, partially tested on movie reviews

---

## Tech Stack

- **Model:** `xlm-roberta-large` (560M params, 100 languages)
- **Framework:** HuggingFace Transformers + PyTorch
- **App:** Streamlit
- **Theme extraction:** Gemini API (`google-genai`)
- **Visualization:** Plotly

---

## LLMs Used (Disclosure)

| LLM | Usage |
|-----|-------|
| Claude | Code scaffolding, error analysis, neutral data generation prompts |
| Gemini | 200 neutral Telugu review generation, theme extraction in app |

---

## Team

**Tech Titans** — IIIT Hyderabad  
Roll No: 2025201080  
Course: Statistical Methods in AI (SMAI) CS7.401  
Academic Year: 2025–26