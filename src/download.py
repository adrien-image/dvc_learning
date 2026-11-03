"""Télécharge les fichiers bruts de MNIST (train et test) dans data/raw."""

from pathlib import Path

from torchvision.datasets import MNIST

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    for train in (True, False):
        MNIST(RAW_DIR, train=train, download=True)


if __name__ == "__main__":
    main()
