from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PREPARED_DIR = ROOT / "data" / "prepared"
MODEL_PATH = ROOT / "models" / "model.pt"

# Statistiques standard de MNIST
MEAN, STD = 0.1307, 0.3081


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


# Paramètres optionnels : valeur utilisée quand la clé est absente du fichier de config.
# Toute clé non listée ici doit être présente dans le fichier.
TRAINING_DEFAULTS = {
    "model": {"hidden_size": 128, "dropout": 0.0},
    "train": {"momentum": 0.9, "weight_decay": 0.0, "seed": 42, "device": "auto"},
}
TRAINING_REQUIRED = {
    "model": {"arch"},
    "train": {"epochs", "batch_size", "lr", "optimizer"},
}


def load_training_config() -> dict:
    """Charge le fichier d'hyperparamètres référencé par `training.config` dans params.yaml,
    complète les clés optionnelles par leur défaut et refuse les clés inconnues (fautes de frappe)."""
    path = load_params()["training"]["config"]
    with open(ROOT / path) as f:
        config = yaml.safe_load(f) or {}

    resolved = {}
    for section, required in TRAINING_REQUIRED.items():
        values = config.get(section) or {}
        defaults = TRAINING_DEFAULTS[section]
        if missing := required - values.keys():
            raise KeyError(f"{path}: clé(s) obligatoire(s) manquante(s) dans '{section}' : {sorted(missing)}")
        if unknown := values.keys() - required - defaults.keys():
            raise KeyError(f"{path}: clé(s) inconnue(s) dans '{section}' : {sorted(unknown)}")
        resolved[section] = defaults | values
    return resolved


def save_split(name: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    """Enregistre un split préparé (images uint8 [N,28,28], labels) dans data/prepared."""
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, PREPARED_DIR / f"{name}.pt")
    print(f"{name}: {len(labels)} images")


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
