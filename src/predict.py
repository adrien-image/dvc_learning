"""Prédictions avec le modèle entraîné.

- Sans image (stage DVC) : `python src/predict.py --split test_qmnist` -> predictions/test_qmnist.csv
  (par défaut --split test, le test set MNIST)
- Avec des images : `python src/predict.py img1.png img2.png ...` -> affiche les prédictions
"""

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import MEAN, MODEL_PATH, ROOT, STD, get_device, load_params, load_split
from model import build_model


def load_model(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(MODEL_PATH, map_location=device)
    mp = ckpt["model_params"]
    model = build_model(mp["arch"], mp["hidden_size"], mp["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, x: torch.Tensor, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    preds, confs = [], []
    for (xb,) in DataLoader(TensorDataset(x), batch_size=batch_size):
        probs = model(xb.to(device)).softmax(1)
        conf, pred = probs.max(1)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
    return torch.cat(preds), torch.cat(confs)


def load_images(paths: list[str]) -> torch.Tensor:
    from PIL import Image, ImageOps

    tensors = []
    for p in paths:
        img = Image.open(p).convert("L").resize((28, 28))
        # MNIST = chiffre clair sur fond sombre : on inverse si le fond est clair
        if np.asarray(img).mean() > 127:
            img = ImageOps.invert(img)
        t = torch.tensor(np.asarray(img), dtype=torch.float32).unsqueeze(0)
        tensors.append(t.div(255).sub(MEAN).div(STD))
    return torch.stack(tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="Images à classer (sinon : le split --split)")
    parser.add_argument("--split", default="test", help="Split de data/prepared à prédire")
    args = parser.parse_args()

    params = load_params()["predict"]
    device = get_device(params["device"])
    model = load_model(device)
    batch_size = params["batch_size"]

    if args.images:
        preds, confs = predict(model, load_images(args.images), batch_size, device)
        for path, pred, conf in zip(args.images, preds, confs):
            print(f"{path}: {pred.item()} ({conf.item():.1%})")
        return

    x, y = load_split(args.split)
    preds, confs = predict(model, x, batch_size, device)
    out = ROOT / "predictions" / f"{args.split}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prediction", "confidence"])
        for i, (label, pred, conf) in enumerate(zip(y.tolist(), preds.tolist(), confs.tolist())):
            writer.writerow([i, label, pred, f"{conf:.4f}"])
    print(f"{len(y)} prédictions écrites dans {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
