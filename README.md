# Qiskit + LLMs Hybrid Project- Integrating BERT and MiniLM Embeddings with Qiskit Variational Circuits for Hybrid Quantum NLP
Ileana Bucur
Course: Large Language Models Zoo, prof.Ca˘grı C¨oltekin

A production-ready skeleton for a **hybrid Quantum + LLM** workflow. 

## What’s inside
- `src/` — Python source (hybrid pipeline, utilities)
- `qiskit/` — quantum circuits, feature maps, VQC/VQE helpers
- `notebooks/` — exploration and demos (start with `00_quickstart.ipynb`)
- `data/` — place your datasets here (ignored in VCS by default)
- `models/` — saved checkpoints and artifacts
- `scripts/` — helper scripts (e.g., `smoke_test.py`)
- `assets/` — images/diagrams

## Quick start (local)
1. **Create env** (recommended)

   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. **Install deps**

   ```bash
   pip install -U -r requirements.txt
   ```
3. **Run a smoke test**

   ```bash
   python scripts/smoke_test.py
   ```
   You should see a JSON summary with Python version, platform, and a sample of project files.

## Quick start (Colab)
Click the badge to open a fresh Colab and attach your drive:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Then run:
```python
!pip install -U qiskit qiskit-machine-learning numpy pandas scikit-learn transformers datasets accelerate torch
```

## Typical workflow
1. **Data prep** → `data/`
2. **Classical text embeddings / LLM** → `src/embeddings.py`
3. **Quantum feature map / circuit** → `qiskit/circuits.py`
4. **Hybrid model training** → `src/train_hybrid.py`
5. **Evaluation & reports** → `notebooks/`

## Reproducibility
- `requirements.txt` pins base libraries commonly used for hybrid QML/NLP.
- For exact results, consider adding a `requirements-lock.txt` produced via `pip freeze` after your first successful run on your machine.

## License
MIT © 2025 Ileana Bucur


See `notebooks/01_demo_sentiment.ipynb` for an end-to-end tiny demo.


### Advanced evaluation (curves & predictions)
```bash
python scripts/evaluate.py       --dataset data/toy_sentiment.csv       --report models/eval_metrics.json       --cm models/confusion_matrix.png       --pr models/pr_curve.png       --roc models/roc_curve.png       --preds models/predictions.csv
```
