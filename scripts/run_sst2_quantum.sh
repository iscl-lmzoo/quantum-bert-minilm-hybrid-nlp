
#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/sst2_quantum
python -m src.train --task sst2 --model bert-base-uncased --head quantum --num_qubits 4 --vqc_reps 2 --batch_size 16 --epochs 3 --outdir results/sst2_quantum/q4_r2_frozen
python -m src.train --task sst2 --model bert-base-uncased --head quantum --num_qubits 6 --vqc_reps 2 --batch_size 16 --epochs 3 --outdir results/sst2_quantum/q6_r2_frozen
