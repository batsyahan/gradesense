"""
GradeSense - Week 2a: Siamese BiLSTM baseline (pre-transformer deep learning).
Run from the project folder:  python week2_bilstm.py        (5way)
                              python week2_bilstm.py 3way   (or 2way)
Scores go to results/results.csv, the model to models/bilstm_<scheme>.pt
"""
import os
import re
import sys
import time
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.data import label_names, load_splits
from src.evaluate import RESULTS_DIR, evaluate_all_splits

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "5way"
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Hyperparameters ----------------
SEED = 42
EMB_DIM = 100      # word embedding size
HIDDEN = 128       # LSTM hidden size per direction
MAX_LEN = 64       # max words per answer (95th pct of student answers is 28)
BATCH = 32
EPOCHS = 25
PATIENCE = 5       # early stopping: stop if val macro-F1 doesn't improve for 5 epochs
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 1. Data + vocabulary ----------------
print(f"\n=== 1. Loading data ({SCHEME}) | device: {device} ===")
splits = load_splits(scheme=SCHEME)
LABELS = label_names(SCHEME)
train = splits["train"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


counter = Counter()
for col in ["reference_answer", "student_answer"]:
    for text in train[col]:
        counter.update(tokenize(text))

# index 0 = padding, index 1 = unknown word
stoi = {w: i + 2 for i, w in enumerate(sorted(counter))}
VOCAB_SIZE = len(stoi) + 2
print(f"Vocabulary: {VOCAB_SIZE} words (built from training answers only)")


def encode(texts):
    ids = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    lens = np.zeros(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        toks = [stoi.get(w, 1) for w in tokenize(text)][:MAX_LEN] or [1]
        ids[i, : len(toks)] = toks
        lens[i] = len(toks)
    return torch.tensor(ids), torch.tensor(lens)


def to_tensors(df):
    r_ids, r_len = encode(list(df["reference_answer"]))
    s_ids, s_len = encode(list(df["student_answer"]))
    return r_ids, r_len, s_ids, s_len


# ---------------- 2. Model ----------------
class Encoder(nn.Module):
    """Embedding -> BiLSTM -> max-pool over words = one vector per sentence."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)

    def forward(self, ids, lens):
        x = self.drop(self.emb(ids))
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        mask = (ids != 0).unsqueeze(-1)
        out = out.masked_fill(~mask, -1e9)
        return out.max(dim=1).values


class SiameseBiLSTM(nn.Module):
    """The SAME encoder reads the reference and the student answer (shared weights).
    The two vectors u, v are combined as [u, v, |u-v|, u*v] and classified."""

    def __init__(self, n_classes):
        super().__init__()
        self.encoder = Encoder()
        d = 2 * HIDDEN
        self.head = nn.Sequential(
            nn.Linear(4 * d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes)
        )

    def forward(self, r_ids, r_len, s_ids, s_len):
        u = self.encoder(r_ids, r_len)
        v = self.encoder(s_ids, s_len)
        return self.head(torch.cat([u, v, torch.abs(u - v), u * v], dim=1))


model = SiameseBiLSTM(len(LABELS)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")


@torch.no_grad()
def predict(df):
    model.eval()
    r_ids, r_len, s_ids, s_len = to_tensors(df)
    preds = []
    for i in range(0, len(df), 256):
        logits = model(r_ids[i:i + 256].to(device), r_len[i:i + 256],
                       s_ids[i:i + 256].to(device), s_len[i:i + 256])
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ---------------- 3. Training ----------------
print("\n=== 2. Training ===")
y_train = train["label"].values
weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_train)
weights = np.clip(weights, None, 10.0)   # cap the huge weight of the tiny non_domain class
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

r_ids, r_len, s_ids, s_len = to_tensors(train)
y = torch.tensor(y_train)
best_f1, best_state, bad_epochs, history = -1.0, None, 0, []

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, total_loss = time.time(), 0.0
    perm = torch.randperm(len(train))
    for i in range(0, len(train), BATCH):
        idx = perm[i:i + BATCH]
        logits = model(r_ids[idx].to(device), r_len[idx], s_ids[idx].to(device), s_len[idx])
        loss = loss_fn(logits, y[idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(idx)

    val_f1 = f1_score(splits["val"]["label"], predict(splits["val"]),
                      labels=list(range(len(LABELS))), average="macro", zero_division=0)
    avg_loss = total_loss / len(train)
    history.append((epoch, avg_loss, val_f1))
    marker = ""
    if val_f1 > best_f1:
        best_f1, bad_epochs, marker = val_f1, 0, "  <- best"
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
    print(f"Epoch {epoch:2d} | loss {avg_loss:.4f} | val macro-F1 {val_f1:.4f} | {time.time() - t0:.1f}s{marker}")
    if bad_epochs >= PATIENCE:
        print(f"Early stopping: no improvement for {PATIENCE} epochs.")
        break

model.load_state_dict(best_state)
torch.save({"state_dict": best_state, "stoi": stoi, "labels": LABELS},
           os.path.join(MODELS_DIR, f"bilstm_{SCHEME}.pt"))
print(f"Best val macro-F1: {best_f1:.4f} (model saved to models/bilstm_{SCHEME}.pt)")

h = np.array(history)
fig, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.plot(h[:, 0], h[:, 1], "o-", label="train loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss")
ax2 = ax1.twinx(); ax2.plot(h[:, 0], h[:, 2], "s-", color="tab:orange", label="val macro-F1")
ax2.set_ylabel("Val macro-F1"); ax1.set_title(f"BiLSTM training - {SCHEME}")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, f"bilstm_training_{SCHEME}.png"), dpi=150); plt.close(fig)

# ---------------- 4. Evaluation ----------------
print("\n=== 3. Evaluation on all splits ===")
evaluate_all_splits("BiLSTM", predict, splits, LABELS, SCHEME)

print("\n=== 4. Macro-F1 comparison so far ===")
results = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
pivot = results[results["scheme"] == SCHEME].pivot(index="model", columns="split", values="macro_f1")
pivot = pivot[[c for c in ["val", "test_ua", "test_uq", "test_ud"] if c in pivot.columns]]
print(pivot.round(3))

# ---------------- 5. The demo example ----------------
print("\n=== 5. Demo example check ===")
demo = pd.DataFrame({
    "reference_answer": ["When the string is tighter, the pitch will be higher."] * 3,
    "student_answer": ["The pitch will be lower.",
                       "A tighter string makes a higher sound.",
                       "The pitch goes up when you tighten it."],
})
for ans, p in zip(demo["student_answer"], predict(demo)):
    print(f"  {ans:45s} -> {LABELS[p]}")

print(f"\nDone. Results in: {RESULTS_DIR}")
