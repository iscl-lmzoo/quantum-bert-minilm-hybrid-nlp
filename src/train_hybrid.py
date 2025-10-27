# src/train_hybrid.py
"""
Training pipeline:
- embed texts -> PCA down to n_qubits
- build EstimatorQNN -> NeuralNetworkClassifier
- train and persist artifacts (pca.pkl, clf.pkl)
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import os
import json
import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import COBYLA

from src.utils.data import load_csv, embed_texts, set_global_seeds
from src.quantum.circuits import make_estimator_qnn

# Optional (simulator)
try:
    from qiskit_aer import AerSimulator
except Exception:
    AerSimulator = None  # type: ignore

def _get_backend(backend: str, seed: int):
    """
    Resolve backend choice. For this project we rely on the EstimatorQNN interface,
    which doesn't require an explicit backend to be passed in. If you want to run
    circuits on a shot-based simulator, you can build a primitive with Aer.
    """
    if backend.lower() in ("aer_simulator", "aersimulator", "aer"):
        if AerSimulator is None:
            raise RuntimeError("qiskit-aer not installed; pip install qiskit-aer")
        return AerSimulator(seed_simulator=seed)
    # Default estimator path (statevector or primitives under the hood)
    return None

def train(dataset_path: str,
          n_qubits: int = 4,
          backend: str = "aer_simulator",
          shots: int = 2048,
          seed: int = 123,
          embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
          models_dir: str = "models") -> Dict[str, Any]:
    """
    Full training routine:
      1) Load CSV (text,label)
      2) Embed with MiniLM/BERT
      3) PCA -> n_qubits
      4) QNN + COBYLA
    Saves PCA and classifier to disk for reproducibility.

    Returns:
        summary dict with shapes and training config.
    """
    os.makedirs(models_dir, exist_ok=True)
    set_global_seeds(seed)

    # 1) Load data
    texts, labels = load_csv(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # 2) Embeddings
    E_train = embed_texts(X_train, model_name=embed_model, seed=seed)
    E_test  = embed_texts(X_test,  model_name=embed_model, seed=seed)

    # 3) Standardize + PCA -> n_qubits
    scaler = StandardScaler(with_mean=True, with_std=True)
    E_train = scaler.fit_transform(E_train)
    E_test  = scaler.transform(E_test)

    pca = PCA(n_components=n_qubits, random_state=seed)
    Z_train = pca.fit_transform(E_train)
    Z_test  = pca.transform(E_test)

    # 4) Quantum Model
    qnn = make_estimator_qnn(num_qubits=n_qubits)
    optimizer = COBYLA(maxiter=200)  # small for demo; can be tuned
    clf = NeuralNetworkClassifier(qnn, optimizer=optimizer, one_hot=False)

    # Fit
    clf.fit(Z_train, y_train)

    # Evaluate quickly to report
    acc = float(clf.score(Z_test, y_test))

    # Persist artifacts
    joblib.dump(pca, os.path.join(models_dir, "pca.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(clf, os.path.join(models_dir, "classifier.pkl"))

    summary = {
        "n_qubits": n_qubits,
        "backend": backend,
        "shots": shots,
        "seed": seed,
        "embed_model": embed_model,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "embedding_dim": int(E_train.shape[1]),
        "pca_dim": n_qubits,
        "test_accuracy": acc,
        "artifacts": ["models/pca.pkl", "models/scaler.pkl", "models/classifier.pkl"],
    }
    with open(os.path.join(models_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
