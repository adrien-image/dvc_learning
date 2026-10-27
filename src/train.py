"""Entraîne le modèle avec les paramètres `model.*` et `train.*` du fichier `training.config`."""

import csv
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import MODEL_PATH, ROOT, get_device, load_split, load_training_config
from model import build_model


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


def make_optimizer(model, p: dict) -> torch.optim.Optimizer:
    if p["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    if p["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=p["lr"], momentum=p["momentum"], weight_decay=p["weight_decay"]
        )
    raise ValueError(f"Optimiseur inconnu : {p['optimizer']!r} (attendu : 'adam' ou 'sgd')")


def main() -> None:
    params = load_training_config()
    mp, tp = params["model"], params["train"]
    torch.manual_seed(tp["seed"])
    device = get_device(tp["device"])
    print(f"Device : {device}")

    train_loader = DataLoader(
        TensorDataset(*load_split("train")), batch_size=tp["batch_size"], shuffle=True
    )
    val_loader = DataLoader(TensorDataset(*load_split("val")), batch_size=1024)

    model = build_model(mp["arch"], mp["hidden_size"], mp["dropout"]).to(device)
    optimizer = make_optimizer(model, tp)

    history = []
    for epoch in range(1, tp["epochs"] + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss, "val_acc": val_acc}
        )
        print(f"epoch {epoch}: train_loss={loss_sum / n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_params": mp}, MODEL_PATH)

    metrics_dir, plots_dir = ROOT / "metrics", ROOT / "plots"
    metrics_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    last = history[-1]
    with open(metrics_dir / "train.json", "w") as f:
        json.dump({"train_loss": last["train_loss"], "val_loss": last["val_loss"], "val_acc": last["val_acc"]}, f, indent=2)
    with open(plots_dir / "train_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
