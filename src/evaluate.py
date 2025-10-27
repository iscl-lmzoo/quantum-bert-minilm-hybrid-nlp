# src/evaluate.py
"""
Evaluation utilities:
- load artifacts
- compute metrics (accuracy, precision, recall, f1)
- export PR/ROC curves + confusion matrix
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_curve, auc,
    classification_report, confusion_matrix
)

from src.utils.data import load_csv, embed_texts, set_global_seeds

def evaluate(dataset_path: str,
             report_path: str = "models/eval_metrics.json",
             cm_path: str = "figs/confusion_matrix.png",
             pr_path: str = "figs/pr_curve.png",
             roc_path: str = "figs/roc_curve.png",
             preds_csv: str = "models/predictions.csv",
             embed_model: Optional[str] = None,
             seed: int = 123) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    os.makedirs(os.path.dirname(pr_path), exist_ok=True)
    os.makedirs(os.path.dirname(roc_path), exist_ok=True)

    set_global_seeds(seed)
    texts, labels = load_csv(dataset_path)

    # Load artifacts
    pca     = joblib.load("models/pca.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    clf     = joblib.load("models/classifier.pkl")

    # If embed model not specified, try to read from summary (nice-to-have)
    if embed_model is None and os.path.exists("models/train_summary.json"):
        with open("models/train_summary.json") as f:
            embed_model = json.load(f).get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    elif embed_model is None:
        embed_model = "sentence-transformers/all-MiniLM-L6-v2"

    E = embed_texts(texts, model_name=embed_model, seed=seed)
    E = scaler.transform(E)
    Z = pca.transform(E)

    # Predict probabilities if available, else decision function
    y_prob = clf.predict_proba(Z) if hasattr(clf, "predict_proba") else clf._forward(Z)  # type: ignore
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y1 = y_prob[:, 1]
    else:
        # fallback for regress-like outputs
        y1 = np.clip(y_prob, 0, 1).ravel()

    y_pred = (y1 >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(labels, y_pred)
    precision, recall, _ = precision_recall_curve(labels, y1)
    fpr, tpr, _ = roc_curve(labels, y1)
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)
    report = classification_report(labels, y_pred, output_dict=True)
    cm = confusion_matrix(labels, y_pred)

    # Save curves
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(True)
    plt.savefig(pr_path, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.grid(True)
    plt.legend()
    plt.savefig(roc_path, bbox_inches="tight"); plt.close()

    # Confusion matrix
    plt.figure()
    plt.imshow(cm, cmap="Blues"); plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xticks([0,1], ["0","1"]); plt.yticks([0,1], ["0","1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(cm_path, bbox_inches="tight"); plt.close()

    # Save predictions
    import pandas as pd
    pd.DataFrame({"text": texts, "label": labels, "pred": y_pred, "prob1": y1}).to_csv(preds_csv, index=False)

    out = {
        "accuracy": float(acc),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "report": report,
        "confusion_matrix": cm.tolist()
    }
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2)
    return out
