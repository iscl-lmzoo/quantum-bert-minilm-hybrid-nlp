import argparse, json
from src.evaluate import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="data/toy_sentiment.csv")
    ap.add_argument("--report", type=str, default="models/eval_metrics.json")
    ap.add_argument("--cm", type=str, default="figs/confusion_matrix.png")
    ap.add_argument("--pr", type=str, default="figs/pr_curve.png")
    ap.add_argument("--roc", type=str, default="figs/roc_curve.png")
    ap.add_argument("--preds-csv", type=str, default="models/predictions.csv")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--embed-model", type=str, default=None)
    args = ap.parse_args()

    res = evaluate(
        dataset_path=args.dataset,
        report_path=args.report,
        cm_path=args.cm,
        pr_path=args.pr,
        roc_path=args.roc,
        preds_csv=args.preds_csv,
        embed_model=args.embed_model,
        seed=args.seed
    )
    print(json.dumps(res, indent=2))
