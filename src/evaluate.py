"""Calcule les métriques de test à partir de predictions/test.csv."""

import csv
import json
from collections import Counter

from common import ROOT


def main() -> None:
    with open(ROOT / "predictions" / "test.csv") as f:
        rows = list(csv.DictReader(f))

    correct = sum(r["label"] == r["prediction"] for r in rows)
    per_class_total = Counter(r["label"] for r in rows)
    per_class_ok = Counter(r["label"] for r in rows if r["label"] == r["prediction"])

    metrics = {
        "test_acc": correct / len(rows),
        "n_errors": len(rows) - correct,
        "per_class_acc": {c: per_class_ok[c] / per_class_total[c] for c in sorted(per_class_total)},
    }
    (ROOT / "metrics").mkdir(exist_ok=True)
    with open(ROOT / "metrics" / "eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Données pour la matrice de confusion (dvc plots)
    (ROOT / "plots").mkdir(exist_ok=True)
    with open(ROOT / "plots" / "confusion.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual", "predicted"])
        writer.writerows((r["label"], r["prediction"]) for r in rows)

    print(f"test_acc = {metrics['test_acc']:.4f} ({metrics['n_errors']} erreurs)")


if __name__ == "__main__":
    main()
