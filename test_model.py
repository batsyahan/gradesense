"""
GradeSense - check the fine-tuned DeBERTa model works on this laptop.
1. Adds DeBERTa's scores to results/results.csv
2. Draws DeBERTa's confusion matrices from the Colab predictions
3. Loads the model on CPU and grades the demo answers
Run:  python test_model.py
"""
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.data import label_names
from src.evaluate import RESULTS_DIR, plot_confusion_matrix

LABELS = label_names("5way")
MODEL_DIR = os.path.join(ROOT, "models", "deberta_5way")

# ---------------- 1. Full results table ----------------
print("\n=== 1. All models, 5-way ===")
res = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
deb = pd.read_csv(os.path.join(RESULTS_DIR, "results_deberta_5way.csv"))
res = res[~((res["model"] == "DeBERTa-v3") & (res["scheme"] == "5way"))]
res = pd.concat([res, deb], ignore_index=True)
res.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
five = res[res["scheme"] == "5way"]
order = ["val", "test_ua", "test_uq", "test_ud"]
for metric in ["macro_f1", "weighted_f1"]:
    print(f"\n{metric}:")
    print(five.pivot(index="model", columns="split", values=metric)[order].round(3))

# ---------------- 2. Confusion matrices ----------------
print("\n=== 2. DeBERTa confusion matrices ===")
for split in ["test_ua", "test_uq", "test_ud"]:
    path = os.path.join(RESULTS_DIR, f"preds_deberta_5way_{split}.csv")
    if not os.path.exists(path):
        print(f"  missing {path} - skipped")
        continue
    p = pd.read_csv(path)
    plot_confusion_matrix(p["label"].values, p["pred"].values, LABELS,
                          title=f"DeBERTa-v3 | 5way | {split}",
                          save_name=f"cm_DeBERTa-v3_5way_{split}.png")
    print(f"  saved cm_DeBERTa-v3_5way_{split}.png")

# ---------------- 3. Load the model on CPU ----------------
print("\n=== 3. Loading the fine-tuned model (CPU) ===")
from transformers import AutoModelForSequenceClassification, AutoTokenizer

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).float().eval()
print(f"Loaded in {time.time() - t0:.1f}s | {sum(p.numel() for p in model.parameters()):,} parameters")


@torch.no_grad()
def grade(question, reference, answer):
    enc = tokenizer(question + " " + reference, answer, truncation=True, max_length=256, return_tensors="pt")
    return torch.softmax(model(**enc).logits.float(), dim=-1)[0].numpy()


# ---------------- 4. Demo ----------------
print("\n=== 4. Demo grading ===")
q = "What happens to the pitch when the string is tightened?"
ref = "When the string is tighter, the pitch will be higher."
answers = [
    "The pitch will be lower.",
    "A tighter string makes a higher sound.",
    "The pitch goes up when you tighten it.",
    "It gets higher.",
    "The string is made of nylon.",
    "I do not know.",
]
times = []
for a in answers:
    t0 = time.time()
    probs = grade(q, ref, a)
    times.append(time.time() - t0)
    i = int(np.argmax(probs))
    print(f"  {a:42s} -> {LABELS[i]:30s} ({probs[i]:.0%})")
print(f"\nAverage time per answer on this laptop: {np.mean(times[1:]):.2f}s")
print("\nModel works locally. Ready for the Streamlit app.")
