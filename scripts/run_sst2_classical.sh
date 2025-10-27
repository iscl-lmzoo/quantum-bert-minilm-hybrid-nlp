
#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/sst2_classical
python -m src.train --task sst2 --model bert-base-uncased --head classical --batch_size 16 --epochs 3 --outdir results/sst2_classical/linear_frozen
