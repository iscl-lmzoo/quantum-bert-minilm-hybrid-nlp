import argparse, json
from src.train_hybrid import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="data/toy_sentiment.csv")
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--backend", type=str, default="aer_simulator")
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    res = train(
        dataset_path=args.dataset,
        n_qubits=args.n_qubits,
        backend=args.backend,
        shots=args.shots,
        seed=args.seed,
        embed_model=args.embed_model
    )
    print(json.dumps(res, indent=2))
