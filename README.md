# Telugu Sentiment Analyzer — T8.3

SMAI Assignment 3 | IIIT Hyderabad 2025–26 | Team: Tech Titans

## Overview
Fine-tuned xlm-roberta-large for Telugu sentiment analysis (Positive / Negative / Neutral).
Achieves 80% overall accuracy and 60% Neutral accuracy on benchmark.

## Models Compared
| Model | Overall | Neutral |
|-------|---------|---------|
| V1 indic-bert + 61 neutral | 70% | 10% |
| V2 indic-bert + 200 neutral | 70% | 10% |
| V3 xlm-roberta-large (final) | 80% | 60% |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Dataset
ai4bharat/IndicSentiment — Telugu config