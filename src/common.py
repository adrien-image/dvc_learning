from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
PREPARED_DIR = ROOT / "data" / "prepared"
MODEL_PATH = ROOT / "models" / "model.pt"

# Statistiques standard de MNIST
MEAN, STD = 0.1307, 0.3081


def load_params(filename: str = "params.yaml") -> dict:
    with open(ROOT / filename) as f:
        return yaml.safe_load(f)


def load_split(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge un split préparé et renvoie (images normalisées [N,1,28,28], labels)."""
    data = torch.load(PREPARED_DIR / f"{name}.pt")
    x = data["images"].float().div(255).sub(MEAN).div(STD).unsqueeze(1)
    return x, data["labels"].long()


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
