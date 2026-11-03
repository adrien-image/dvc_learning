# MNIST + DVC

Pipeline DVC pour entraîner un classifieur MNIST (PyTorch) et faire des prédictions,
avec les hyperparamètres d'entraînement dans `training_config.yaml` et le suivi des essais via `dvc exp`.

## Installation

```bash
uv sync
source .venv/bin/activate   # ou préfixer chaque commande par `uv run`
```

Pour partager données et modèles via un stockage S3 : voir [Stockage S3](#stockage-s3-remote-dvc).

## Pipeline (`dvc.yaml`)

| Stage                 | Script                        | Dépend de (données)                           | Params                          | Sorties                                                            |
|-----------------------|-------------------------------|-----------------------------------------------|---------------------------------|--------------------------------------------------------------------|
| `download_mnist`      | `src/download.py mnist`       | –                                             | –                               | `data/raw/MNIST`                                                   |
| `download_qmnist`     | `src/download.py qmnist`      | –                                             | –                               | `data/raw/QMNIST`                                                  |
| `prepare_train`       | `src/prepare_train.py`        | `data/raw/MNIST`                              | `prepare`                       | `data/prepared/train.pt`, `data/prepared/val.pt`                   |
| `prepare_test`        | `src/prepare_test.py`         | `data/raw/MNIST`                              | –                               | `data/prepared/test.pt`                                            |
| `prepare_test_qmnist` | `src/prepare_test_qmnist.py`  | `data/raw/QMNIST`                             | –                               | `data/prepared/test_qmnist.pt`                                     |
| `train`               | `src/train.py`                | `train.pt`, `val.pt`                          | `training.config` + son contenu | `models/model.pt`, `metrics/train.json`, `plots/train_history.csv` |
| `predict@test`        | `src/predict.py --split test` | `models/model.pt`, `test.pt`                  | `predict`                       | `predictions/test.csv`                                             |
| `predict@test_qmnist` | `src/predict.py --split test_qmnist` | `models/model.pt`, `test_qmnist.pt`    | `predict`                       | `predictions/test_qmnist.csv`                                      |
| `evaluate`            | `src/evaluate.py test test_qmnist` | `predictions/test.csv`, `predictions/test_qmnist.csv` | –                 | `metrics/eval.json`, `plots/confusion_{test,test_qmnist}.csv`      |

```
download_mnist ──┬─► prepare_train ──► train ──┬───────────────────────┐
                 │                             ▼                       ▼
                 └─► prepare_test ───────► predict@test       predict@test_qmnist
                                               │                       ▲     │
download_qmnist ──► prepare_test_qmnist ───────┼───────────────────────┘     │
                                               ▼                             │
                                           evaluate ◄────────────────────────┘
```

La préparation du train set et celle des test sets sont séparées : changer `prepare.val_size`
ou `prepare.seed` relance `prepare_train` et la suite, mais pas `prepare_test` ni
`prepare_test_qmnist`. Les test sets ne dépendent d'aucun paramètre et restent identiques
d'une expérience à l'autre.

### Deux test sets : MNIST et QMNIST

- **`test`** : le test set officiel de MNIST (10 000 chiffres).
- **`test_qmnist`** : [QMNIST](https://github.com/facebookresearch/qmnist) `test50k`, les
  50 000 chiffres que MNIST avait tirés du même fonds NIST mais jamais publiés
  (Yadav & Bottou, 2019). Même source et même prétraitement que MNIST, et aucun modèle n'a
  jamais été réglé dessus : un écart entre `test.acc` et `test_qmnist.acc` signale un
  surapprentissage du test set officiel.

`predict` est un stage `foreach` : un sous-stage `predict@<split>` par test set.
`metrics/eval.json` contient une entrée par test set :

```json
{"test":        {"acc": 0.98, "n_errors": 157, "n_samples": 10000, "per_class_acc": {...}},
 "test_qmnist": {"acc": 0.98, "n_errors": 926, "n_samples": 50000, "per_class_acc": {...}}}
```

Pour ajouter un autre test set : un stage qui produit `data/prepared/<split>.pt`, le nom du split
dans la liste `foreach` de `predict`, et dans la commande et les deps de `evaluate`.

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

### Surcharger un paramètre optionnel avec `dvc exp` (`+`, `++`, `~`)

`dvc exp run -S` ne modifie pas la config en mémoire : il **réécrit le fichier YAML** de
l'expérience, en suivant la syntaxe d'override de [Hydra](https://hydra.cc/docs/advanced/override_grammar/basic/).
Hydra est strict : sans préfixe, `-S` ne peut que **modifier une clé qui existe déjà** dans le
fichier. Pour un paramètre optionnel absent du fichier, il faut le dire explicitement avec un préfixe :

| Syntaxe                        | Clé présente dans le fichier | Clé absente du fichier |
|--------------------------------|------------------------------|------------------------|
| `train.x=1`                    | modifiée                     | ❌ erreur              |
| `+train.x=1`                   | ❌ erreur                    | ajoutée                |
| `++train.x=1`                  | modifiée                     | ajoutée                |
| `~train.x`                     | supprimée (→ défaut du code) | ❌ erreur              |

En cas d'erreur, DVC affiche `Invalid --set-param value`. Penser aux guillemets autour de
l'argument (`'...'`), pour que le shell n'interprète pas `~`.

**Exemple : ajouter un paramètre optionnel `train.label_smoothing`**

1. Le déclarer avec sa valeur par défaut dans `src/common.py` :

   ```python
   TRAINING_DEFAULTS = {
       "model": {"hidden_size": 128, "dropout": 0.0},
       "train": {"momentum": 0.9, "weight_decay": 0.0, "seed": 42, "device": "auto",
                 "label_smoothing": 0.0},
   }
   ```

2. L'utiliser dans `src/train.py` :

   ```python
   loss = nn.functional.cross_entropy(model(x), y, label_smoothing=tp["label_smoothing"])
   ```

3. Commiter. `training_config.yaml` n'a pas besoin de changer : sans la clé, le défaut `0.0`
   s'applique et le comportement est identique à avant.

4. Lancer des expériences en ajoutant la clé avec `+` (elle n'est pas dans le fichier) :

   ```bash
   # une expérience
   dvc exp run -n smooth-0.1 -S 'training_config.yaml:+train.label_smoothing=0.1'

   # une grille : `+` se combine avec une liste de valeurs (ici 3 runs)
   dvc exp run --queue -S 'training_config.yaml:+train.label_smoothing=0.05,0.1,0.2'
   dvc queue start

   # forme qui marche que la clé soit présente ou non
   dvc exp run -S 'training_config.yaml:++train.label_smoothing=0.1'
   ```

5. Comparer :

   ```bash
   dvc exp show --keep 'Experiment|label_smoothing|^test.acc|^test_qmnist.acc' --drop '.*'
   ```

   La colonne `train.label_smoothing` n'apparaît renseignée que pour les expériences où la clé
   a été écrite. Pour les autres, elle est vide (`-`), et la valeur réellement utilisée est le
   défaut défini dans le code.

À l'inverse, pour revenir au défaut d'une clé présente dans le fichier :

```bash
dvc exp run -S 'training_config.yaml:~train.momentum'
```

Si `dvc exp apply` est appliqué à une expérience lancée avec `+`, la clé est ajoutée à
`training_config.yaml` : commiter le fichier la rend alors permanente.

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
dvc exp show --only-changed --sort-by metrics/eval.json:test.acc --sort-order desc
dvc exp show --keep 'Experiment|^test.acc|^test_qmnist.acc' --drop '.*'   # MNIST vs QMNIST
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

## Stockage S3 (remote DVC)

Les données, le modèle et les prédictions sont dans le cache DVC local. Pour les partager ou
les sauvegarder, le projet fournit un S3 local ([SeaweedFS](https://github.com/seaweedfs/seaweedfs),
compatible S3) via Docker Compose, et un script qui y connecte DVC.

### Lancer le S3

```bash
cp .env.example .env              # puis changer S3_SECRET_ACCESS_KEY
docker compose up -d --wait       # démarre le S3 sur http://localhost:9000 et crée le bucket
```

`.env` contient les identifiants (`S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`), le bucket
(`S3_BUCKET`, défaut `dvc-storage`) et le port (`S3_PORT`, défaut `9000`). Il n'est pas versionné.
Seule l'API S3 est exposée, et uniquement sur `127.0.0.1`. Les données sont dans le volume Docker `s3-data`.

```bash
docker compose ps                 # état
docker compose logs -f s3         # logs
docker compose down               # arrêter (les données sont conservées)
docker compose down -v            # arrêter et supprimer les données
```

### Connecter DVC

```bash
./scripts/setup_dvc_remote.sh     # ou : uv run scripts/setup_dvc_remote.sh
```

Le script lit `.env` et configure le remote par défaut `s3` :

- `.dvc/config` (versionné) : `url = s3://dvc-storage`, `endpointurl = http://localhost:9000`
- `.dvc/config.local` (ignoré par git) : `access_key_id`, `secret_access_key`

À relancer après un changement de `.env`.

### Envoyer / récupérer

```bash
dvc push                          # envoie les sorties du workspace (dvc.lock courant)
dvc push --all-commits            # ... et celles de tous les commits
dvc pull                          # récupère les sorties correspondant au dvc.lock courant
dvc status --cloud                # compare le cache local et le remote
```

Sur une autre machine (ou un nouveau clone) : `uv sync`, copier `.env`, lancer
`./scripts/setup_dvc_remote.sh`, puis `dvc pull`. Pas besoin de relancer le pipeline.

Les expériences (`dvc exp`) sont stockées dans des refs git. Pour les partager avec leurs
artefacts, il faut un remote git en plus : `dvc exp push origin <exp>`.

### Utiliser un vrai bucket AWS S3

```bash
dvc remote add --default --force s3 s3://mon-bucket/dvc
dvc remote modify s3 --unset endpointurl
```

Les identifiants viennent alors de la config AWS habituelle (`~/.aws/credentials`, `AWS_PROFILE`,
variables d'environnement), ou de `dvc remote modify --local s3 access_key_id ...`.

## Prédire sur ses propres images

```bash
python src/predict.py chemin/vers/chiffre.png autre.jpg
```

Les images sont converties en niveaux de gris 28×28 (inversées automatiquement si le fond est clair).
