
#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/legal_quantum
python -m src.train --task legal --model bert-base-uncased --head quantum --num_qubits 4 --vqc_reps 2 --batch_size 8 --epochs 5 --outdir results/legal_quantum/q4_r2_frozen
