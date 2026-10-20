"""
GradeSense - data loading utilities.

Dataset: SciEntsBank (SemEval-2013 Task 7), via Hugging Face: nkazi/SciEntsBank
Columns: id, question, reference_answer, student_answer, label
Splits : train, test_ua (unseen answers), test_uq (unseen questions), test_ud (unseen domains)
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_NAME = "nkazi/SciEntsBank"
TEST_SPLITS = ["test_ua", "test_uq", "test_ud"]

# 5-way labels, in the dataset's own id order
LABELS_5WAY = [
    "correct",
    "contradictory",
    "partially_correct_incomplete",
    "irrelevant",
    "non_domain",
]

# How each 5-way label collapses into the 3-way and 2-way schemes
LABEL_SCHEMES = {
    "5way": {name: name for name in LABELS_5WAY},
    "3way": {
        "correct": "correct",
        "contradictory": "contradictory",
        "partially_correct_incomplete": "incorrect",
        "irrelevant": "incorrect",
        "non_domain": "incorrect",
    },
    "2way": {
        "correct": "correct",
        "contradictory": "incorrect",
        "partially_correct_incomplete": "incorrect",
        "irrelevant": "incorrect",
        "non_domain": "incorrect",
    },
}

SCHEME_LABELS = {
    "5way": LABELS_5WAY,
    "3way": ["correct", "contradictory", "incorrect"],
    "2way": ["correct", "incorrect"],
}


def load_splits(scheme: str = "5way", val_size: float = 0.1, seed: int = 42):
    """
    Load SciEntsBank as pandas DataFrames.

    Returns a dict with keys: train, val, test_ua, test_uq, test_ud.
    Each DataFrame has columns: id, question, reference_answer, student_answer,
    label_name (string) and label (int id within the chosen scheme).

    'val' is a stratified slice of the official train split, used for model
    selection, so the three official test splits stay untouched until the end.
    """
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"scheme must be one of {list(LABEL_SCHEMES)}")

    raw = load_dataset(DATASET_NAME)
    label_names_5 = raw["train"].features["label"].names
    mapping = LABEL_SCHEMES[scheme]
    names = SCHEME_LABELS[scheme]
    name_to_id = {n: i for i, n in enumerate(names)}

    def to_df(split):
        df = raw[split].to_pandas()
        df["label_name"] = df["label"].map(lambda i: mapping[label_names_5[i]])
        df["label"] = df["label_name"].map(name_to_id)
        return df

    full_train = to_df("train")
    train_df, val_df = train_test_split(
        full_train,
        test_size=val_size,
        stratify=full_train["label"],
        random_state=seed,
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
    }
    for split in TEST_SPLITS:
        splits[split] = to_df(split)
    return splits


def label_names(scheme: str = "5way"):
    return SCHEME_LABELS[scheme]
