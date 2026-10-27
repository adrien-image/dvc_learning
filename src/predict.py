"""Prédictions avec le modèle entraîné.

- Sans argument (stage DVC) : prédit sur le test set MNIST -> predictions/test.csv
- Avec des images : `python src/predict.py img1.png img2.png ...` -> affiche les prédictions
"""

import argparse
import csv

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
        if sum(img.getdata()) / (28 * 28) > 127:
            img = ImageOps.invert(img)
        t = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 28, 28)
        tensors.append(t.div(255).sub(MEAN).div(STD))
    return torch.stack(tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="Images à classer (sinon : test set MNIST)")
    args = parser.parse_args()

    params = load_params()
    device = get_device(params["train"]["device"])
    model = load_model(device)
    batch_size = params["predict"]["batch_size"]

    if args.images:
        preds, confs = predict(model, load_images(args.images), batch_size, device)
        for path, pred, conf in zip(args.images, preds, confs):
            print(f"{path}: {pred.item()} ({conf.item():.1%})")
        return

    x, y = load_split("test")
    preds, confs = predict(model, x, batch_size, device)
    out = ROOT / "predictions" / "test.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prediction", "confidence"])
        for i, (label, pred, conf) in enumerate(zip(y.tolist(), preds.tolist(), confs.tolist())):
            writer.writerow([i, label, pred, f"{conf:.4f}"])
    print(f"{len(y)} prédictions écrites dans {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
