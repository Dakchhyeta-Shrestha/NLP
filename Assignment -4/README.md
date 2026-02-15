# Do You Agree? – Sentence-BERT NLI Project

This project implements a **Natural Language Inference (NLI)** system that predicts whether two sentences show **Entailment**, **Neutral**, or **Contradiction**. The system follows a **Sentence-BERT (SBERT) style Siamese architecture** and was developed as part of an NLP assignment.

---

## Project Overview

The project consists of two main stages:

1. **BERT Pretraining with Masked Language Modeling (MLM)**  
   A small BERT model is trained from scratch on a subset of the English Wikipedia dataset using the MLM objective.

2. **Sentence-BERT Fine-Tuning for NLI**  
   The trained encoder is reused in a Siamese architecture and fine-tuned on the SNLI dataset using Softmax (cross-entropy) loss to classify sentence pairs.

---

## Datasets Used

- **Wikipedia (English)** – used for Masked Language Modeling (subset only).
- **SNLI (Stanford Natural Language Inference)** – used for entailment, neutral, and contradiction classification.

Both datasets are accessed via the Hugging Face Datasets library.

---

## Model & Training

- Custom lightweight BERT trained from scratch  
- Mean pooling for sentence embeddings  
- Softmax loss for multi-class NLI classification  
- Evaluation using precision, recall, F1-score, and accuracy  

---

## Evaluation Results

| Class | Precision | Recall | F1-Score | Support |
|------|----------|--------|---------|---------|
| Entailment | 0.66 | 0.64 | 0.65 | 2047 |
| Neutral | 0.59 | 0.64 | 0.62 | 1941 |
| Contradiction | 0.62 | 0.59 | 0.60 | 2012 |
| **Accuracy** |  |  | **0.62** | 6000 |
| **Macro Avg** | 0.62 | 0.62 | 0.62 | 6000 |
| **Weighted Avg** | 0.62 | 0.62 | 0.62 | 6000 |

---

## Web Application

A **Streamlit web application** is included to demonstrate real-time inference.

Run the app with:
```bash
streamlit run app.py
```

---

## Output screenshots

### Contradiction
Below is the output screenshot of Contradiction
![Contradiction](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-4/outputs/contradiction.png)

### Entailment
Below is the output screenshot of Entailment
![Entailment](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-4/outputs/entailment.jpeg)

### Neutral
Below is the output screenshot of Neutral
![Neutral](https://github.com/Dakchhyeta-Shrestha/NLP/blob/main/Assignment%20-4/outputs/neutral.png)

---

## Notes

Due to time and computational constraints, training was performed on reduced dataset sizes and limited epochs.

---
