# MNIST + DVC

Pipeline DVC pour entraîner un classifieur MNIST (PyTorch) et faire des prédictions,
avec les hyperparamètres d'entraînement dans `training_config.yaml` et le suivi des essais via `dvc exp`.

## Installation

```bash
uv sync
source .venv/bin/activate   # ou préfixer chaque commande par `uv run`
```

## Pipeline (`dvc.yaml`)

| Stage      | Script               | Params      | Sorties                                               |
|------------|----------------------|-------------|-------------------------------------------------------|
| `prepare`  | `src/prepare.py`     | `prepare`   | `data/raw`, `data/prepared/{train,val,test}.pt`       |
| `train`    | `src/train.py`       | `training.config` + son contenu | `models/model.pt`, `metrics/train.json`, `plots/train_history.csv` |
| `predict`  | `src/predict.py`     | `predict`   | `predictions/test.csv`                                |
| `evaluate` | `src/evaluate.py`    | –           | `metrics/eval.json`, `plots/confusion.csv`            |

```bash
dvc repro          # ne relance que les stages dont les deps/params ont changé
dvc dag            # visualiser le graphe
```

## Paramètres

`params.yaml` désigne le fichier d'hyperparamètres via `training.config`
(par défaut `training_config.yaml`). `src/train.py` lit ce chemin et charge le fichier,
et `dvc.yaml` le référence par interpolation (`${training.config}`) : le nom du fichier
n'est écrit en dur nulle part ailleurs.

Contenu du fichier de config d'entraînement :

- `model.arch` (`cnn` | `mlp`), `model.hidden_size`, `model.dropout`
- `train.epochs`, `train.batch_size`, `train.lr`, `train.optimizer` (`adam` | `sgd`),
  `train.momentum`, `train.weight_decay`, `train.seed`, `train.device` (`auto` | `cpu` | `mps` | `cuda`)

### Paramètres obligatoires et optionnels

`src/common.py` définit `TRAINING_REQUIRED` (clés qui doivent figurer dans le fichier) et
`TRAINING_DEFAULTS` (clés optionnelles et leur valeur par défaut). Une clé obligatoire
manquante ou une clé inconnue (faute de frappe) fait échouer l'entraînement.

Pour ajouter un nouveau paramètre optionnel : l'ajouter à `TRAINING_DEFAULTS`, puis l'utiliser
dans `train.py`. Les configs existantes restent valides sans modification.

Avec `dvc exp`, une clé absente du fichier s'ajoute avec `+` (syntaxe Hydra) et se retire avec `~` :

```bash
dvc exp run -S 'training_config.yaml:+train.weight_decay=0.0001'   # clé absente du fichier
dvc exp run -S 'training_config.yaml:~train.momentum'              # retour à la valeur par défaut
```

`dvc exp show` n'affiche que les valeurs écrites dans le fichier : une clé absente apparaît
vide (`-`), et la valeur réellement utilisée est le défaut défini dans le code.

`params.yaml` (le reste) : `training.config`, `prepare.val_size`, `prepare.seed`, `predict.batch_size`, `predict.device`

Modifier un paramètre puis `dvc repro` ne relance que ce qui en dépend
(ex. changer `train.lr` ne retélécharge pas les données).

## Expériences avec `dvc exp`

```bash
# Les params hors params.yaml se préfixent par leur fichier : -S <fichier>:<clé>=<valeur>
# Une expérience, en surchargeant des params sans toucher au fichier
dvc exp run -n mlp-sgd \
  -S training_config.yaml:model.arch=mlp \
  -S training_config.yaml:train.optimizer=sgd \
  -S training_config.yaml:train.lr=0.05

# Grille de recherche : produit cartésien des valeurs (ici 3 x 2 = 6 runs)
dvc exp run --queue \
  -S training_config.yaml:train.lr=0.0005,0.001,0.003 \
  -S training_config.yaml:train.epochs=3,5
# Utiliser un autre fichier de config (ex. configs/mlp.yaml, à créer et commiter)
dvc exp run -S training.config=configs/mlp.yaml

dvc queue start -j 2        # lance la file avec 2 workers
dvc queue status

# Comparer
dvc exp show
dvc exp show --only-changed --sort-by metrics/eval.json:test_acc --sort-order desc
dvc exp diff <exp1> <exp2>
dvc plots diff <exp1> <exp2> --open   # courbes d'apprentissage + matrice de confusion

# Garder la meilleure
dvc exp apply <exp>          # applique params + résultats dans le workspace
git commit -am "..."         # ou : dvc exp branch <exp> <branche>

# Nettoyer
dvc exp remove <exp>         # ou --all-commits / -A pour tout supprimer
```

⚠️ `dvc exp run` (sans `--queue`) **applique ses résultats au workspace** : `training_config.yaml`
contient alors les valeurs de l'expérience. Une expérience mise en file ensuite part de cet
état modifié. Pour repartir de la version commitée : `git checkout training_config.yaml dvc.lock && dvc checkout`.

## Prédire sur ses propres images

```bash
python src/predict.py chemin/vers/chiffre.png autre.jpg
```

Les images sont converties en niveaux de gris 28×28 (inversées automatiquement si le fond est clair).
