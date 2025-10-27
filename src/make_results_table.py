
"""
GENERATE LaTeX TABLE FROM results/summary.csv
- Best accuracy per (task, model, head, num_qubits, vqc_reps).
"""
from __future__ import annotations
import argparse, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    g = df.groupby(['task','model','head','num_qubits','vqc_reps']).acc_test.max().reset_index()

    lines = [
        '\\begin{table}[t]','\\centering','\\small',
        '\\begin{tabular}{l l l r r r}','\\toprule',
        'Task & Model & Head & Qubits & Reps & Acc \\\\','\\midrule'
    ]
    for _, r in g.iterrows():
        q = int(r['num_qubits']) if r['head']=='quantum' else 0
        reps = int(r['vqc_reps']) if r['head']=='quantum' else 0
        lines.append(f"{r['task']} & {r['model']} & {r['head']} & {q} & {reps} & {r['acc_test']:.3f} \\")
    lines += ['\\bottomrule','\\end{tabular}','\\caption{Best accuracy per configuration.}','\\label{tab:results}','\\end{table}']

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f: f.write('\n'.join(lines))

if __name__ == '__main__':
    main()
