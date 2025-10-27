
"""
TRAINING LOOP (PyTorch)
- Tasks: 'sst2' (GLUE) and 'legal' (tiny demo).
- Logs a summary CSV and saves the best checkpoint.
"""
from __future__ import annotations
import argparse, os, time, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_decay
import evaluate

from data import load_sst2, load_legal_binary
from models import HybridModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['sst2','legal'], default='sst2')
    ap.add_argument('--model', type=str, default='bert-base-uncased')
    ap.add_argument('--head', choices=['classical','quantum'], default='quantum')
    ap.add_argument('--fine_tune_bert', action='store_true')
    ap.add_argument('--num_qubits', type=int, default=4)
    ap.add_argument('--vqc_reps', type=int, default=2)
    ap.add_argument('--max_length', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--legal_root', type=str, default='data/legal_mini')
    ap.add_argument('--outdir', type=str, default='results/run')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def prepare_data(args):
    if args.task == 'sst2':
        ds = load_sst2()
        text_key = 'sentence'; label_key = 'label'
    else:
        ds = load_legal_binary(args.legal_root)
        ds = ds.map(lambda ex: {'sentence': ex['text'], 'label': ex['label']})
        text_key = 'sentence'; label_key = 'label'
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ds = ds.map(lambda ex: tok(ex[text_key], truncation=True, max_length=args.max_length), batched=True)
    ds.set_format(type='torch', columns=['input_ids','attention_mask', label_key])
    return ds, label_key

def train_loop(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds, label_key = prepare_data(args)

    model = HybridModel(model_name=args.model, num_classes=2, head=args.head,
                        fine_tune_bert=args.fine_tune_bert, num_qubits=args.num_qubits, vqc_reps=args.vqc_reps).to(device)

    train_loader = DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds['validation'], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds['test'], batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_decay(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    metric = evaluate.load('accuracy')

    os.makedirs(args.outdir, exist_ok=True)
    best_val = -1.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            labels = batch[label_key].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()

        model.eval()
        preds, refs = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attn = batch['attention_mask'].to(device)
                labels = batch[label_key].to(device)
                logits = model(input_ids=input_ids, attention_mask=attn)
                p = torch.argmax(logits, dim=-1)
                preds.extend(p.cpu().tolist()); refs.extend(labels.cpu().tolist())
        val_acc = metric.compute(predictions=preds, references=refs)['accuracy']
        print(f"Epoch {epoch} | train_loss={running/len(train_loader):.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best.pt'))

    train_time_s = time.time() - start

    model.load_state_dict(torch.load(os.path.join(args.outdir, 'best.pt'), map_location=device))
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            labels = batch[label_key].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn)
            p = torch.argmax(logits, dim=-1)
            preds.extend(p.cpu().tolist()); refs.extend(labels.cpu().tolist())
    test_acc = metric.compute(predictions=preds, references=refs)['accuracy']

    summary_csv = 'results/summary.csv'
    header = ['task','model','head','fine_tune_bert','num_qubits','vqc_reps','max_length','batch','epochs','acc_test','train_time_s']
    row = [args.task, args.model, args.head, int(args.fine_tune_bert), args.num_qubits, args.vqc_reps, args.max_length, args.batch_size, args.epochs, test_acc, train_time_s]
    write_header = not os.path.exists(summary_csv)
    with open(summary_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

    print("Test accuracy:", test_acc)
    print(f"Summary row appended to {summary_csv}")

if __name__ == '__main__':
    args = parse_args()
    train_loop(args)
