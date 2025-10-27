#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_sst2_classical.sh
bash scripts/run_sst2_quantum.sh
bash scripts/run_legal_binary_quantum.sh
python -m src.plots --results_csv results/summary.csv --outdir results
python -m src.make_results_table --csv results/summary.csv --out paper/tables/results.tex
