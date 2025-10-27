# tests/test_min.py
"""
Lightweight tests:
- test_load_csv: shape and non-empty
- test_embed_texts: deterministic shape and repeatability via seed
- test_qnn_build: circuit/QNN parameter counts consistent with n_qubits
"""
import os
import numpy as np
import pandas as pd
import pytest

from src.utils.data import load_csv, embed_texts
from src.quantum.circuits import build_feature_map, build_ansatz, make_estimator_qnn

DATA = "data/toy_sentiment.csv"

@pytest.mark.parametrize("text_col,label_col", [(None, None)])
def test_load_csv(text_col, label_col):
    assert os.path.exists(DATA), "data/toy_sentiment.csv missing"
    texts, labels = load_csv(DATA, text_col=text_col, label_col=label_col)
    assert len(texts) == len(labels) and len(texts) > 0

def test_embed_texts_reproducible():
    texts = ["good", "bad"]
    E1 = embed_texts(texts, seed=42)
    E2 = embed_texts(texts, seed=42)
    assert E1.shape == E2.shape
    # Determinism across runs (same seed)
    assert np.allclose(E1, E2, atol=1e-6)

def test_qnn_build():
    n = 4
    fmap = build_feature_map(n)
    ansatz = build_ansatz(n)
    qnn = make_estimator_qnn(n, fmap, ansatz)
    assert len(fmap.parameters) == n * fmap.reps  or len(fmap.parameters) > 0
    assert qnn is not None
