import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# 🔹 Path to your model
MODEL_PATH = "D:\6th Sem\CI\CI-PROJECT\CI Project\local\roberta_train\train4_artifacts\checkpoints\best_epoch_2"

# 🔹 Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.eval()

# 🔹 Load your validation dataset again
# (you must use same dev.tsv)
import pandas as pd

def read_tsv(path):
    return pd.read_csv(path, sep='\t', header=None, names=['text','labels','id'])

dev_df = read_tsv("dev.tsv")  # make sure this file exists

# 🔹 Convert labels
NUM_CLASSES = 28

def encode_labels(s):
    v = np.zeros(NUM_CLASSES)
    for x in str(s).split():
        if x.isdigit():
            v[int(x)] = 1
    return v

dev_df["label_vec"] = dev_df["labels"].apply(encode_labels)

# 🔹 Inference
all_probs = []
all_labels = []

for _, row in tqdm(dev_df.iterrows(), total=len(dev_df)):
    inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)
    
    all_probs.append(probs.numpy()[0])
    all_labels.append(row["label_vec"])

y_scores = np.array(all_probs)
y_true = np.array(all_labels)

# 🔹 Flatten
y_scores = y_scores.ravel()
y_true = y_true.ravel()

# 🔹 ROC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 🔹 PR
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# 🔹 Plot ROC
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 🔹 Plot PR
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()