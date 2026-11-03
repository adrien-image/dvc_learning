"""Convertit le test set brut de MNIST en split test."""

from torchvision.datasets import MNIST

from common import RAW_DIR, save_split


def main() -> None:
    mnist = MNIST(RAW_DIR, train=False)
    save_split("test", mnist.data, mnist.targets)


if __name__ == "__main__":
    main()
