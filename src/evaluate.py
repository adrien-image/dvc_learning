"""Calcule les métriques de chaque test set à partir de predictions/<split>.csv.

    python src/evaluate.py test test_qmnist
    -> metrics/eval.json : {"test": {...}, "test_qmnist": {...}}
    -> plots/confusion_<split>.csv
"""

import argparse
import csv
import json
from collections import Counter

from common import ROOT


def evaluate_split(split: str) -> dict:
    with open(ROOT / "predictions" / f"{split}.csv") as f:
        rows = list(csv.DictReader(f))

    correct = sum(r["label"] == r["prediction"] for r in rows)
    per_class_total = Counter(r["label"] for r in rows)
    per_class_ok = Counter(r["label"] for r in rows if r["label"] == r["prediction"])

    # Données pour la matrice de confusion (dvc plots)
    (ROOT / "plots").mkdir(exist_ok=True)
    with open(ROOT / "plots" / f"confusion_{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual", "predicted"])
        writer.writerows((r["label"], r["prediction"]) for r in rows)

    return {
        "acc": correct / len(rows),
        "n_errors": len(rows) - correct,
        "n_samples": len(rows),
        "per_class_acc": {c: per_class_ok[c] / per_class_total[c] for c in sorted(per_class_total)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("splits", nargs="+", help="Splits à évaluer (predictions/<split>.csv)")
    args = parser.parse_args()

    metrics = {split: evaluate_split(split) for split in args.splits}
    (ROOT / "metrics").mkdir(exist_ok=True)
    with open(ROOT / "metrics" / "eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for split, m in metrics.items():
        print(f"{split}: acc = {m['acc']:.4f} ({m['n_errors']} erreurs / {m['n_samples']})")


if __name__ == "__main__":
    main()
