"""
GradeSense - baseline models, in a form that can be saved, reloaded and used for live comparison.

All baselines expose predict_proba(questions, references, answers) -> array (n, n_labels).

- TfidfBaseline : TF-IDF similarity features (+ answer TF-IDF for the "full" variant) -> logistic regression
                  (identical features to week1_baseline.py)
- SbertBaseline : frozen Sentence-BERT embeddings of reference and answer -> logistic regression ("no fine-tuning")
- BiLSTMBaseline: loads the Siamese BiLSTM trained by week2_bilstm.py (models/bilstm_<scheme>.pt)
"""
import re

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _clean(t):
    return re.sub(r"[^a-z0-9 ]", " ", str(t).lower())


def _full_proba(clf, X, n_labels):
    """predict_proba with a column for every label id, even if a class was absent in training."""
    p = clf.predict_proba(X)
    out = np.zeros((p.shape[0], n_labels))
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return out


# ---------------------------------------------------------------- TF-IDF
class TfidfBaseline:
    def __init__(self, variant="full", n_labels=5):
        assert variant in ("full", "sim")
        self.variant = variant
        self.n_labels = n_labels
        self.vec = None
        self.scaler = None
        self.clf = None

    @property
    def name(self):
        return "TFIDF-Full" if self.variant == "full" else "TFIDF-Sim"

    def _sim_features(self, q, r, a):
        q, r, a = [list(map(_clean, x)) for x in (q, r, a)]
        R, S, Q = self.vec.transform(r), self.vec.transform(a), self.vec.transform(q)
        cos_rs = np.asarray(R.multiply(S).sum(axis=1)).ravel()   # rows are L2-normalised, so dot = cosine
        cos_qs = np.asarray(Q.multiply(S).sum(axis=1)).ravel()
        feats = []
        for rr, ss in zip(r, a):
            rw, sw = set(rr.split()), set(ss.split())
            feats.append([len(rw & sw) / max(len(rw), 1), len(sw) / max(len(rw), 1), len(sw)])
        return np.column_stack([cos_rs, cos_qs, np.array(feats).reshape(-1, 3)])

    def _X(self, q, r, a):
        sim = self.scaler.transform(self._sim_features(q, r, a))
        if self.variant == "sim":
            return sim
        return hstack([csr_matrix(sim), self.vec.transform([_clean(x) for x in a])]).tocsr()

    def fit(self, q, r, a, y):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
        self.vec.fit([_clean(x) for x in list(r) + list(a) + list(q)])
        self.scaler = StandardScaler().fit(self._sim_features(q, r, a))
        if self.variant == "full":
            self.clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)
        else:
            self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.clf.fit(self._X(q, r, a), np.asarray(y))
        return self

    def predict_proba(self, q, r, a):
        return _full_proba(self.clf, self._X(q, r, a), self.n_labels)

    def save(self, path):
        joblib.dump({"kind": "tfidf", "variant": self.variant, "n_labels": self.n_labels,
                     "vec": self.vec, "scaler": self.scaler, "clf": self.clf}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        m = cls(d["variant"], d["n_labels"])
        m.vec, m.scaler, m.clf = d["vec"], d["scaler"], d["clf"]
        return m


# ---------------------------------------------------------------- SBERT (frozen) + LR
class SbertBaseline:
    name = "SBERT-LR"

    def __init__(self, sbert_name="sentence-transformers/all-MiniLM-L6-v2", n_labels=5, encoder=None):
        self.sbert_name = sbert_name
        self.n_labels = n_labels
        self.clf = None
        self._encoder = encoder

    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.sbert_name, device="cpu")
        return self._encoder

    def _embed(self, texts):
        texts = [str(t) for t in texts]
        uniq = list(dict.fromkeys(texts))
        emb = self.encoder().encode(uniq, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
        lookup = {t: emb[i] for i, t in enumerate(uniq)}
        return np.stack([lookup[t] for t in texts])

    def _X(self, r, a):
        u, v = self._embed(r), self._embed(a)
        cos = (u * v).sum(axis=1, keepdims=True)
        return np.hstack([cos, np.abs(u - v), u * v])

    def fit(self, q, r, a, y):
        self.clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        self.clf.fit(self._X(r, a), np.asarray(y))
        return self

    def predict_proba(self, q, r, a):
        return _full_proba(self.clf, self._X(r, a), self.n_labels)

    def save(self, path):
        joblib.dump({"kind": "sbert", "sbert_name": self.sbert_name, "n_labels": self.n_labels,
                     "clf": self.clf}, path)

    @classmethod
    def load(cls, path, encoder=None):
        d = joblib.load(path)
        m = cls(d["sbert_name"], d["n_labels"], encoder=encoder)
        m.clf = d["clf"]
        return m


# ---------------------------------------------------------------- Siamese BiLSTM
def _build_bilstm_classes():
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class Encoder(nn.Module):
        """Embedding -> BiLSTM -> max-pool (same attribute names as week2_bilstm.py so checkpoints load)."""

        def __init__(self, vocab_size, emb_dim=100, hidden=128):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
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
        def __init__(self, vocab_size, n_classes, emb_dim=100, hidden=128):
            super().__init__()
            self.encoder = Encoder(vocab_size, emb_dim, hidden)
            d = 2 * hidden
            self.head = nn.Sequential(nn.Linear(4 * d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))

        def forward(self, r_ids, r_len, s_ids, s_len):
            u = self.encoder(r_ids, r_len)
            v = self.encoder(s_ids, s_len)
            return self.head(torch.cat([u, v, torch.abs(u - v), u * v], dim=1))

    return torch, SiameseBiLSTM


class BiLSTMBaseline:
    name = "BiLSTM"
    MAX_LEN = 64

    def __init__(self, model, stoi, labels, torch_module):
        self.model, self.stoi, self.labels, self.torch = model, stoi, labels, torch_module

    @classmethod
    def load(cls, path):
        torch, SiameseBiLSTM = _build_bilstm_classes()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        stoi, labels = ckpt["stoi"], list(ckpt["labels"])
        model = SiameseBiLSTM(len(stoi) + 2, len(labels))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model, stoi, labels, torch)

    def _encode(self, texts):
        torch = self.torch
        ids = np.zeros((len(texts), self.MAX_LEN), dtype=np.int64)
        lens = np.zeros(len(texts), dtype=np.int64)
        for i, text in enumerate(texts):
            toks = [self.stoi.get(w, 1) for w in re.findall(r"[a-z0-9]+", str(text).lower())][:self.MAX_LEN] or [1]
            ids[i, :len(toks)] = toks
            lens[i] = len(toks)
        return torch.tensor(ids), torch.tensor(lens)

    def predict_proba(self, q, r, a):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(a), 256):
                r_ids, r_len = self._encode(list(r)[i:i + 256])
                s_ids, s_len = self._encode(list(a)[i:i + 256])
                out.append(torch.softmax(self.model(r_ids, r_len, s_ids, s_len), dim=-1).numpy())
        return np.concatenate(out)
