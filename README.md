# Transformer-Based Multi-Label Emotion Classification (Comparative Study)

## Overview

This project presents a comparative study of transformer-based models for multi-label emotion classification using the GoEmotions dataset. The objective is to analyze how well modern contextual language models capture nuanced human emotions in text, including ambiguity and multi-emotion scenarios.

The study evaluates three transformer architectures:

* BERT
* RoBERTa
* XLNet

---

## Problem Statement

Emotion detection in text is inherently complex due to overlapping emotions, contextual dependencies, and subtle linguistic cues. This project addresses the challenge of identifying multiple emotions simultaneously from a single text input.

---

## Dataset

* **GoEmotions Dataset (Google)**
* Contains 28 emotion categories
* Multi-label classification problem

---

## Models Used

* **BERT** – Baseline transformer model
* **RoBERTa** – Optimized training strategy with improved performance
* **XLNet** – Permutation-based transformer for better context modeling

---

## Methodology

1. Data preprocessing and tokenization
2. Fine-tuning transformer models on multi-label classification
3. Generating predictions for validation set
4. Evaluating performance using:

   * Micro Precision, Recall, F1-score
   * Macro Precision, Recall, F1-score
5. Comparative analysis across all models

---

## Evaluation Metrics

* Micro Precision / Recall / F1
* Macro Precision / Recall / F1
* Per-class (per-emotion) F1 scores
* ROC and Precision-Recall analysis

---

## Key Results

| Model   | Micro F1 | Macro F1 |
| ------- | -------- | -------- |
| BERT    | ~0.59    | ~0.53    |
| RoBERTa | ~0.60    | ~0.53    |
| XLNet   | ~0.59    | ~0.53    |

* **RoBERTa achieved the best overall performance**
* BERT performed competitively as a strong baseline
* XLNet showed stable but slightly lower performance

---

## Insights

* RoBERTa better captures contextual nuances in frequent and ambiguous emotions
* BERT performs well on lexically clear emotional expressions
* XLNet provides consistent but not superior results in this setup
* Class imbalance impacts performance across rare emotions

---

## Project Structure

```plaintext
emotion-detection/
│
├── notebook/
│   └── comparative_analysis.ipynb
│
├── src/
│   └── roc_pr.py
│
├── results/
│   ├── final_report.json
│   ├── training_metrics.xlsx
│   ├── per_class_metrics.xlsx
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn

---

## Learning Outcomes

* Implemented multi-label classification using transformers
* Compared performance of different transformer architectures
* Analyzed model behavior across multiple emotion categories
* Understood challenges of class imbalance in NLP tasks

---
