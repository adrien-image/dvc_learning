"""Télécharge MNIST et produit les splits train / val / test en tenseurs uint8."""

import torch
from torchvision.datasets import MNIST

from common import PREPARED_DIR, ROOT, load_params


def main() -> None:
    params = load_params()["prepare"]
    raw_dir = ROOT / "data" / "raw"

    train = MNIST(raw_dir, train=True, download=True)
    test = MNIST(raw_dir, train=False, download=True)

    gen = torch.Generator().manual_seed(params["seed"])
    perm = torch.randperm(len(train.data), generator=gen)
    val_idx, train_idx = perm[: params["val_size"]], perm[params["val_size"] :]

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": (train.data[train_idx], train.targets[train_idx]),
        "val": (train.data[val_idx], train.targets[val_idx]),
        "test": (test.data, test.targets),
    }
    for name, (images, labels) in splits.items():
        torch.save({"images": images, "labels": labels}, PREPARED_DIR / f"{name}.pt")
        print(f"{name}: {len(labels)} images")


if __name__ == "__main__":
    main()
