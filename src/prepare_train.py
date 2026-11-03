"""Découpe le train set brut de MNIST en splits train / val (paramètres `prepare.*`)."""

import torch
from torchvision.datasets import MNIST

from common import RAW_DIR, load_params, save_split


def main() -> None:
    params = load_params()["prepare"]
    mnist = MNIST(RAW_DIR, train=True)

    gen = torch.Generator().manual_seed(params["seed"])
    perm = torch.randperm(len(mnist.data), generator=gen)
    val_idx, train_idx = perm[: params["val_size"]], perm[params["val_size"] :]

    save_split("train", mnist.data[train_idx], mnist.targets[train_idx])
    save_split("val", mnist.data[val_idx], mnist.targets[val_idx])


if __name__ == "__main__":
    main()
