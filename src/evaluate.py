"""
GradeSense - shared evaluation module.
Every model is scored with these same functions, so comparisons are fair.
Predictions are saved per split (results/preds_<model>_<scheme>_<split>.csv) for error analysis.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def compute_metrics(y_true, y_pred, label_names):
    labels = list(range(len(label_names)))
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=label_names,
                                           output_dict=True, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, label_names, title, save_name=None):
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(1.4 * len(label_names) + 3, 1.2 * len(label_names) + 2))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=label_names,
                yticklabels=label_names, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    if save_name:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(RESULTS_DIR, save_name), dpi=150)
    plt.close(fig)
    return fig


def save_predictions(model_name, scheme, split, ids, y_true, y_pred):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"preds_{model_name}_{scheme}_{split}.csv")
    pd.DataFrame({"id": list(ids), "label": list(y_true), "pred": list(y_pred)}).to_csv(path, index=False)
    return path


def evaluate_all_splits(model_name, predict_fn, splits, label_names, scheme, plot=True, save_preds=True):
    """predict_fn(df) -> array of label ids. Scores every split, saves predictions,
    and records the scores in results/results.csv."""
    rows = []
    for split in ["val", "test_ua", "test_uq", "test_ud"]:
        if split not in splits:
            continue
        df = splits[split]
        y_pred = np.asarray(predict_fn(df))
        m = compute_metrics(df["label"].values, y_pred, label_names)
        rows.append({"model": model_name, "scheme": scheme, "split": split,
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"]})
        if save_preds and "id" in df.columns:
            save_predictions(model_name, scheme, split, df["id"].values, df["label"].values, y_pred)
        if plot and split != "val":
            plot_confusion_matrix(df["label"].values, y_pred, label_names,
                                  title=f"{model_name} | {scheme} | {split}",
                                  save_name=f"cm_{model_name}_{scheme}_{split}.png".replace(" ", "_"))
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    save_results(summary)
    return summary


def save_results(summary_df):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "results.csv")
    if os.path.exists(path):
        old = pd.read_csv(path)
        if len(old):
            keys = set(zip(summary_df["model"], summary_df["scheme"]))
            old = old[[(m, s) not in keys for m, s in zip(old["model"], old["scheme"])]]
            summary_df = pd.concat([old, summary_df], ignore_index=True)
    summary_df.to_csv(path, index=False)


def save_json(obj, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)
