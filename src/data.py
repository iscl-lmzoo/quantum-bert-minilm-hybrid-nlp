# src/utils/data.py
"""
Data utilities: CSV loader and text embedding helpers.
- load_csv: robust CSV loader with column auto-detection
- embed_texts: sentence-transformers embeddings with deterministic seeds
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import random
import numpy as np
import pandas as pd

@dataclass
class DatasetSplits:
    X_train: List[str]
    y_train: np.ndarray
    X_val: List[str]
    y_val: np.ndarray
    X_test: List[str]
    y_test: np.ndarray

def set_global_seeds(seed: int) -> None:
    """Make runs deterministic where possible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional in this project
        pass

def load_csv(path: str,
             text_col: Optional[str] = None,
             label_col: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
    """
    Load a simple CSV dataset with text + label columns.
    If columns are not specified, tries common names.

    Returns:
        texts: list[str]
        labels: np.ndarray of shape (N,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    # Heuristics if user didn't specify columns
    text_candidates = [text_col] if text_col else ["text", "sentence", "review"]
    label_candidates = [label_col] if label_col else ["label", "y", "target", "sentiment"]

    tc = next((c for c in text_candidates if c in df.columns), None)
    lc = next((c for c in label_candidates if c in df.columns), None)
    if tc is None or lc is None:
        raise ValueError(f"Could not infer columns. CSV has: {list(df.columns)}")

    texts = df[tc].astype(str).tolist()
    labels = df[lc].to_numpy()
    return texts, labels

def embed_texts(texts: List[str],
                model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                seed: int = 123) -> np.ndarray:
    """
    Encode texts into dense vectors using sentence-transformers.
    The model is frozen; we only use it as a feature extractor.

    Args:
        texts: list of raw strings
        model_name: HF model id (e.g., MiniLM or BERT variants)
        seed: seed for internal RNGs (tokenizer order, etc.)

    Returns:
        embeddings: np.ndarray (N, D)
    """
    set_global_seeds(seed)
    # Lazy import to avoid heavy startup when not needed
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    # Deterministic batching
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return embeddings
