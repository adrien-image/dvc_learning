"""Convertit QMNIST test50k (les 50 000 chiffres du test set NIST non publiés dans MNIST)
en split test_qmnist."""

from torchvision.datasets import QMNIST

from common import RAW_DIR, save_split


def main() -> None:
    qmnist = QMNIST(RAW_DIR, what="test50k")
    # targets contient 8 colonnes de métadonnées : la première est le chiffre
    save_split("test_qmnist", qmnist.data, qmnist.targets[:, 0])


if __name__ == "__main__":
    main()
