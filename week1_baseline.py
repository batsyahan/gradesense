"""
GradeSense - Week 1: data exploration + TF-IDF baselines.
Run from the project folder:  python week1_baseline.py
All plots and scores are saved in results/
"""
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import label_names, load_splits
from src.evaluate import RESULTS_DIR, evaluate_all_splits

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "5way"   # 5way / 3way / 2way
os.makedirs(RESULTS_DIR, exist_ok=True)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

# ---------------- 1. Load data ----------------
print(f"\n=== 1. Loading SciEntsBank ({SCHEME}) ===")
splits = load_splits(scheme=SCHEME)
LABELS = label_names(SCHEME)
train = splits["train"]
for name, df in splits.items():
    print(f"{name:8s}: {len(df):5d} rows | {df['question'].nunique():3d} unique questions")

# ---------------- 2. Label distribution ----------------
print("\n=== 2. Label distribution (%) ===")
dist = pd.DataFrame({
    name: df["label_name"].value_counts(normalize=True).reindex(LABELS).fillna(0)
    for name, df in splits.items()
})
print((dist * 100).round(1))
ax = dist.T.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="tab10")
ax.set_ylabel("Proportion"); ax.set_title("Label distribution per split")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "label_distribution.png"), dpi=150); plt.close()

# ---------------- 3. Lengths (sets max_len for DeBERTa) ----------------
print("\n=== 3. Input lengths ===")
train_words = train["student_answer"].str.split().str.len()
print("Student answer words  - mean %.1f, 95th pct %d" % (train_words.mean(), np.percentile(train_words, 95)))
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    pair_lens = [
        len(tok(q + " " + r, s)["input_ids"])
        for q, r, s in zip(train["question"], train["reference_answer"], train["student_answer"])
    ]
    print("Full input tokens     - mean %.1f, 95th pct %d, max %d"
          % (np.mean(pair_lens), np.percentile(pair_lens, 95), max(pair_lens)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(train_words, bins=40); axes[0].set_title("Student answer length (words)")
    axes[1].hist(pair_lens, bins=40); axes[1].set_title("Full input length (BERT tokens)")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "length_distribution.png"), dpi=150); plt.close()
except Exception as e:
    print("Tokenizer step skipped:", e)

# ---------------- 4. TF-IDF baselines ----------------
print("\n=== 4. TF-IDF baselines ===")

def clean(t):
    return re.sub(r"[^a-z0-9 ]", " ", str(t).lower())

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
vec.fit(pd.concat([train["reference_answer"], train["student_answer"], train["question"]]).map(clean))

def sim_features(df):
    R = vec.transform(df["reference_answer"].map(clean))
    S = vec.transform(df["student_answer"].map(clean))
    Q = vec.transform(df["question"].map(clean))
    cos_rs = np.asarray(R.multiply(S).sum(axis=1)).ravel()   # rows are L2-normalised, so dot = cosine
    cos_qs = np.asarray(Q.multiply(S).sum(axis=1)).ravel()
    feats = []
    for r, s in zip(df["reference_answer"].map(clean), df["student_answer"].map(clean)):
        rw, sw = set(r.split()), set(s.split())
        feats.append([len(rw & sw) / max(len(rw), 1), len(sw) / max(len(rw), 1), len(sw)])
    return np.column_stack([cos_rs, cos_qs, np.array(feats)])

scaler = StandardScaler().fit(sim_features(train))
X_sim = lambda df: scaler.transform(sim_features(df))
X_full = lambda df: hstack([csr_matrix(X_sim(df)), vec.transform(df["student_answer"].map(clean))]).tocsr()

print("\n--- TFIDF-Sim (similarity features only) ---")
clf_sim = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_sim(train), train["label"])
evaluate_all_splits("TFIDF-Sim", lambda df: clf_sim.predict(X_sim(df)), splits, LABELS, SCHEME)

print("\n--- TFIDF-Full (similarity + answer words) ---")
clf_full = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0).fit(X_full(train), train["label"])
evaluate_all_splits("TFIDF-Full", lambda df: clf_full.predict(X_full(df)), splits, LABELS, SCHEME)

# ---------------- 5. Comparison ----------------
print("\n=== 5. Macro-F1 comparison ===")
results = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
pivot = results[results["scheme"] == SCHEME].pivot(index="model", columns="split", values="macro_f1")
pivot = pivot[[c for c in ["val", "test_ua", "test_uq", "test_ud"] if c in pivot.columns]]
print(pivot.round(3))
pivot.drop(columns=["val"], errors="ignore").T.plot(kind="bar", figsize=(7, 3.5))
plt.ylabel("Macro-F1"); plt.title(f"Baselines - {SCHEME}"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f"baselines_{SCHEME}.png"), dpi=150); plt.close()

# ---------------- 6. Error analysis ----------------
print("\n=== 6. Sample errors on test_ua (TFIDF-Sim) ===")
ua = splits["test_ua"].copy()
ua["pred"] = [LABELS[i] for i in clf_sim.predict(X_sim(ua))]
wrong = ua[ua["pred"] != ua["label_name"]]
print(f"{len(wrong)} / {len(ua)} wrong")
sample = wrong.sample(min(10, len(wrong)), random_state=1)
sample[["reference_answer", "student_answer", "label_name", "pred"]].to_csv(
    os.path.join(RESULTS_DIR, f"errors_tfidf_{SCHEME}.csv"), index=False)
for _, r in sample.iterrows():
    print(f"\nREF : {r['reference_answer']}\nANS : {r['student_answer']}\nTRUE: {r['label_name']}  |  PRED: {r['pred']}")

print(f"\nDone. Plots and scores saved in: {RESULTS_DIR}")
