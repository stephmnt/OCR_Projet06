---
title: OCR Projet 06
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OCR Projet 06 – Crédit

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stephmnt/credit-scoring-mlops/deploy.yml)](https://github.com/stephmnt/credit-scoring-mlops/actions/workflows/deploy.yml)
[![GitHub Release Date](https://img.shields.io/github/release-date/stephmnt/credit-scoring-mlops?display_date=published_at&style=flat-square)](https://github.com/stephmnt/credit-scoring-mlops/releases)
[![project_license](https://img.shields.io/github/license/stephmnt/credit-scoring-mlops.svg)](https://github.com/stephmnt/credit-scoring-mlops/blob/main/LICENSE)

## Lancer MLFlow

Le notebook est configure pour utiliser un serveur MLflow local (`http://127.0.0.1:5000`).
Pour voir les runs et creer l'experiment, demarrer le serveur avec le meme backend :

```shell
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri "file:${PWD}/mlruns" \
  --default-artifact-root "file:${PWD}/mlruns"
```

Seulement l'interface (sans API), lancer :

```shell
mlflow ui --backend-store-uri "file:${PWD}/mlruns" --port 5000
```

Pour tester le serving du modele en staging :

```shell
mlflow models serve -m "models:/credit_scoring_model/Staging" -p 5001 --no-conda
```

## API FastAPI

L'API attend un payload JSON avec une cle `data`. La valeur peut etre un objet unique (un client) ou une liste d'objets (plusieurs clients). La liste des features requises (jeu reduit) est disponible via l'endpoint `/features`. Les autres champs sont optionnels et seront completes par des valeurs par defaut.

Inputs minimums (10 + `SK_ID_CURR`) derives d'une selection par correlation (voir `/features`) :

- `EXT_SOURCE_2`
- `EXT_SOURCE_3`
- `AMT_ANNUITY`
- `EXT_SOURCE_1`
- `CODE_GENDER`
- `DAYS_EMPLOYED`
- `AMT_CREDIT`
- `AMT_GOODS_PRICE`
- `DAYS_BIRTH`
- `FLAG_OWN_CAR`

Parametres utiles (selection des features) :

- `FEATURE_SELECTION_METHOD` (defaut: `correlation`)
- `FEATURE_SELECTION_TOP_N` (defaut: `8`)
- `FEATURE_SELECTION_MIN_CORR` (defaut: `0.02`)

### Environnement Poetry (recommande)

Le fichier `pyproject.toml` fixe des versions compatibles pour un stack recent
(`numpy>=2`, `pyarrow>=15`, `scikit-learn>=1.6`). L'environnement vise Python
3.11.

```shell
poetry env use 3.11
poetry install
poetry run pytest -q
poetry run uvicorn app.main:app --reload --port 7860
```

Important : le modele `HistGB_final_model.pkl` doit etre regenere avec la
nouvelle version de scikit-learn (re-execution de
`P6_MANET_Stephane_notebook_modélisation.ipynb`, cellule de sauvegarde pickle).

Note : `requirements.txt` est aligne sur `pyproject.toml` (meme versions).

### Exemple d'input (schema + valeurs)

Schema :

```json
{
  "data": {
    "SK_ID_CURR": "int",
    "EXT_SOURCE_2": "float",
    "EXT_SOURCE_3": "float",
    "AMT_ANNUITY": "float",
    "EXT_SOURCE_1": "float",
    "CODE_GENDER": "str",
    "DAYS_EMPLOYED": "int",
    "AMT_CREDIT": "float",
    "AMT_GOODS_PRICE": "float",
    "DAYS_BIRTH": "int",
    "FLAG_OWN_CAR": "str"
  }
}
```

Valeurs d'exemple :

```json
{
  "data": {
    "SK_ID_CURR": 100002,
    "EXT_SOURCE_2": 0.61,
    "EXT_SOURCE_3": 0.75,
    "AMT_ANNUITY": 24700.5,
    "EXT_SOURCE_1": 0.45,
    "CODE_GENDER": "M",
    "DAYS_EMPLOYED": -637,
    "AMT_CREDIT": 406597.5,
    "AMT_GOODS_PRICE": 351000.0,
    "DAYS_BIRTH": -9461,
    "FLAG_OWN_CAR": "N"
  }
}
```

Note : l'API valide strictement les champs requis (`/features`). Pour afficher
toutes les colonnes possibles : `/features?include_all=true`.

### Demo live (commandes cles en main)

Lancer l'API :

```shell
uvicorn app.main:app --reload --port 7860
```

Verifier le service (HF) :

```shell
BASE_URL="https://stephmnt-credit-scoring-mlops.hf.space"
curl -s "${BASE_URL}/health"
```

Voir les features attendues (HF) :

```shell
curl -s "${BASE_URL}/features"
```

Predire un client (HF) :

```shell
curl -s -X POST "${BASE_URL}/predict?threshold=0.5" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "SK_ID_CURR": 100002,
      "EXT_SOURCE_2": 0.61,
      "EXT_SOURCE_3": 0.75,
      "AMT_ANNUITY": 24700.5,
      "EXT_SOURCE_1": 0.45,
      "CODE_GENDER": "M",
      "DAYS_EMPLOYED": -637,
      "AMT_CREDIT": 406597.5,
      "AMT_GOODS_PRICE": 351000.0,
      "DAYS_BIRTH": -9461,
      "FLAG_OWN_CAR": "N"
    }
  }'
```

Predire plusieurs clients (batch, HF) :

```shell
curl -s -X POST "${BASE_URL}/predict?threshold=0.45" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "SK_ID_CURR": 100002,
        "EXT_SOURCE_2": 0.61,
        "EXT_SOURCE_3": 0.75,
        "AMT_ANNUITY": 24700.5,
        "EXT_SOURCE_1": 0.45,
        "CODE_GENDER": "M",
        "DAYS_EMPLOYED": -637,
        "AMT_CREDIT": 406597.5,
        "AMT_GOODS_PRICE": 351000.0,
        "DAYS_BIRTH": -9461,
        "FLAG_OWN_CAR": "N"
      },
      {
        "SK_ID_CURR": 100003,
        "EXT_SOURCE_2": 0.52,
        "EXT_SOURCE_3": 0.64,
        "AMT_ANNUITY": 19000.0,
        "EXT_SOURCE_1": 0.33,
        "CODE_GENDER": "F",
        "DAYS_EMPLOYED": -1200,
        "AMT_CREDIT": 320000.0,
        "AMT_GOODS_PRICE": 280000.0,
        "DAYS_BIRTH": -12000,
        "FLAG_OWN_CAR": "Y"
      }
    ]
  }'
```

Exemple d'erreur (champ requis manquant, HF) :

```shell
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "EXT_SOURCE_2": 0.61
    }
  }'
```

## Monitoring & Data Drift (Etape 3)

L'API enregistre les appels `/predict` en JSONL (inputs, outputs, latence).
Par defaut, les logs sont stockes dans `logs/predictions.jsonl`.

Variables utiles :

- `LOG_PREDICTIONS=1` active l'ecriture des logs (defaut: 1)
- `LOG_DIR=logs`
- `LOG_FILE=predictions.jsonl`
- `LOGS_ACCESS_TOKEN` pour proteger l'endpoint `/logs`
- `LOG_HASH_SK_ID=1` pour anonymiser `SK_ID_CURR`

Exemple local :

```shell
LOG_PREDICTIONS=1 LOG_DIR=logs uvicorn app.main:app --reload --port 7860
```

Recuperer les logs (HF) :

Configurer `LOGS_ACCESS_TOKEN` dans les secrets du Space, puis :

```shell
curl -s -H "X-Logs-Token: $LOGS_ACCESS_TOKEN" "${BASE_URL}/logs?tail=200"
```

Alternative :

```shell
curl -s -H "Authorization: Bearer $LOGS_ACCESS_TOKEN" "${BASE_URL}/logs?tail=200"
```

Apres quelques requêtes, gélérer le rapport de drift :

```shell
python monitoring/drift_report.py \
  --logs logs/predictions.jsonl \
  --reference data/data_final.parquet \
  --output-dir reports
```

Le rapport HTML est généré dans `reports/drift_report.html` (avec des plots dans
`reports/plots/`). Sur Hugging Face, le disque est éphemère : télécharger les logs
avant d'analyser.

Le rapport inclut aussi la distribution des scores predits et le taux de prediction
(option `--score-bins` pour ajuster le nombre de bins).

Captures (snapshot local du reporting + stockage):

- Rapport: `docs/monitoring/drift_report.html` + `docs/monitoring/plots/`
- Stockage des logs: `docs/monitoring/logs_storage.png`

## Contenu de la release

- **Preparation + pipeline** : nettoyage / preparation, encodage, imputation et pipeline d'entrainement presentes.
- **Gestion du desequilibre** : un sous-echantillonnage est applique sur le jeu d'entrainement final.
- **Comparaison multi-modeles** : baseline, Naive Bayes, Logistic Regression, Decision Tree, Random Forest,
  HistGradientBoosting, LGBM, XGB sont compares.
- **Validation croisee + tuning** : `StratifiedKFold`, `GridSearchCV` et Hyperopt sont utilises.
- **Score metier + seuil optimal** : le `custom_score` est la metrique principale des tableaux de comparaison et de la CV, avec un `best_threshold` calcule.
- **Explicabilite** : feature importance, SHAP et LIME sont inclus.
- **Selection de features par correlation** : top‑N numeriques + un petit set categoriel, expose via `/features`.
- **Monitoring & drift** : rapport HTML avec KS/PSI + distribution des scores predits et taux de prediction
  (snapshots dans `docs/monitoring/`).
- **CI/CD** : tests avec couverture (`pytest-cov`), build Docker et deploy vers Hugging Face Spaces.

![Screenshot MLFlow](https://raw.githubusercontent.com/stephmnt/credit-scoring-mlops/main/screen-mlflow.png)

### Manques prioritaires

* Mission 2 Étape 4 non couverte: pas de profiling/optimisation post‑déploiement ni rapport de gains, à livrer avec une version optimisée.

### Preuves / doc à compléter

* Lien explicite vers le dépôt public + stratégie de versions/branches à ajouter dans README.md.
* Preuve de model registry/serving MLflow à conserver (capture UI registry ou commande de serving) en plus de screen-mlflow.png.
* Dataset de référence non versionné (data_final.parquet est ignoré), documenter l’obtention pour exécuter drift_report.py.
* Badge GitHub Actions pointe vers OCR_Projet05 dans README.md, corriger l’URL.
* RGPD/PII: LOG_HASH_SK_ID est désactivé par défaut dans main.py, préciser l’activation en prod dans README.md.

### Améliorations recommandées

* Compléter les tests API: /logs (auth OK/KO), batch predict, param threshold, SK_ID_CURR manquant, outliers dans test_api.py.
* Simplifier le fallback ALLOW_MISSING_ARTIFACTS et DummyModel si les artefacts sont versionnés (nettoyer main.py et conftest.py).
* Unifier la gestion des dépendances (Poetry vs requirements.txt) et aligner pyproject.toml / requirements.txt.
* Si l’évaluateur attend une stratégie de branches, créer une branche feature et fusionner pour preuve.
