.PHONY: venv install train eval test clean

ifeq ($(OS),Windows_NT)
PY_SYS=python
VENV_BIN=.venv/Scripts
PY=$(VENV_BIN)/python.exe
PIP=$(VENV_BIN)/pip.exe
else
PY_SYS=python3
VENV_BIN=.venv/bin
PY=$(VENV_BIN)/python
PIP=$(VENV_BIN)/pip
endif

venv:
	$(PY_SYS) -m venv .venv
	$(PY) -m pip install -U pip

install:
	$(PIP) install -U -r requirements.txt || true

train:
	$(PY) scripts/train.py --dataset data/toy_sentiment.csv --n-qubits 4 --backend aer_simulator --shots 2048 --seed 123

eval:
	$(PY) scripts/evaluate.py --dataset data/toy_sentiment.csv --report models/eval_metrics.json --cm figs/confusion_matrix.png --pr figs/pr_curve.png --roc figs/roc_curve.png

test:
	$(PIP) install -U pytest
	$(PY) -m pytest -q 
