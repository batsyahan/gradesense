"""
GradeSense - build every baseline artifact the web app needs.

  1. TF-IDF baselines (Sim and Full): refit (deterministic, same features as week1), save, evaluate
  2. Siamese BiLSTM: load models/bilstm_<scheme>.pt (no retraining), evaluate
  3. SBERT (no fine-tuning) + logistic regression: train, save, evaluate

Each model's predictions are saved to results/preds_<model>_<scheme>_<split>.csv (for Error Analysis)
and its scores to results/results.csv.

Run from the project folder:  python scripts/build_baselines.py        (5way)
"""
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.baselines import BiLSTMBaseline, SbertBaseline, TfidfBaseline
from src.data import label_names, load_splits
from src.evaluate import RESULTS_DIR, evaluate_all_splits

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "5way"
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def cols(df):
    return list(df["question"]), list(df["reference_answer"]), list(df["student_answer"])


def predictor(model):
    return lambda df: model.predict_proba(*cols(df)).argmax(axis=1)


print(f"\n=== Loading SciEntsBank ({SCHEME}) ===")
splits = load_splits(scheme=SCHEME)
LABELS = label_names(SCHEME)
train = splits["train"]
q, r, a = cols(train)
y = train["label"].values

for variant in ("sim", "full"):
    t0 = time.time()
    m = TfidfBaseline(variant, len(LABELS)).fit(q, r, a, y)
    m.save(os.path.join(MODELS_DIR, f"tfidf_{variant}_{SCHEME}.joblib"))
    print(f"\n--- {m.name} (fit in {time.time() - t0:.1f}s) ---")
    evaluate_all_splits(m.name, predictor(m), splits, LABELS, SCHEME)

bilstm_path = os.path.join(MODELS_DIR, f"bilstm_{SCHEME}.pt")
if os.path.exists(bilstm_path):
    print("\n--- BiLSTM (loaded from checkpoint, not retrained) ---")
    m = BiLSTMBaseline.load(bilstm_path)
    evaluate_all_splits("BiLSTM", predictor(m), splits, LABELS, SCHEME)
else:
    print(f"\nSkipping BiLSTM: {bilstm_path} not found (run week2_bilstm.py first).")

print("\n--- SBERT (no fine-tuning) + LR: encoding answers, ~1-3 minutes on CPU ---")
t0 = time.time()
m = SbertBaseline(n_labels=len(LABELS)).fit(q, r, a, y)
m.save(os.path.join(MODELS_DIR, f"sbert_lr_{SCHEME}.joblib"))
print(f"Trained in {time.time() - t0:.1f}s")
evaluate_all_splits("SBERT-LR", predictor(m), splits, LABELS, SCHEME)

print("\n=== Macro-F1, all models ===")
res = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
sub = res[res["scheme"] == SCHEME]
order = [s for s in ["val", "test_ua", "test_uq", "test_ud"] if s in set(sub["split"])]
print(sub.pivot(index="model", columns="split", values="macro_f1")[order].round(3))
print("\nDone. Saved models in models/ and predictions in results/.")
