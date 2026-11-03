"""Télécharge les fichiers bruts d'un dataset dans data/raw/<DATASET>.

    python src/download.py mnist     # MNIST train + test   -> data/raw/MNIST
    python src/download.py qmnist    # QMNIST test (60k)    -> data/raw/QMNIST
"""

import argparse
from pathlib import Path

from torchvision.datasets import MNIST, QMNIST

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mnist", "qmnist"])
    args = parser.parse_args()

    if args.dataset == "mnist":
        for train in (True, False):
            MNIST(RAW_DIR, train=train, download=True)
    else:
        QMNIST(RAW_DIR, what="test50k", download=True)


if __name__ == "__main__":
    main()
