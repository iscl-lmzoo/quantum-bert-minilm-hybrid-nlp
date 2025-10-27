
"""
PLOTS FROM results/summary.csv
- Creates bar plots comparing 'classical' vs 'quantum' heads per task.
"""
from __future__ import annotations
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_csv', required=True)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.results_csv)

    for task, sub in df.groupby('task'):
        piv = sub.groupby(['model','head']).acc_test.max().unstack('head')
        ax = piv.plot(kind='bar')
        ax.set_title(f'Accuracy by head — {task}'); ax.set_ylabel('Accuracy')
        fig = ax.get_figure()
        fig.savefig(os.path.join(args.outdir, f'acc_by_head_{task}.png'), dpi=200)
        plt.close(fig)

if __name__ == '__main__':
    main()
