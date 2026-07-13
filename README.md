# Fraud Detection System

Fraud classifier built on the IEEE-CIS Fraud Detection dataset, with a
time-based validation split, cost-based threshold selection, and a served
API endpoint.

## Setup (Windows / Anaconda)

```
conda create -n fraud-detection python=3.10
conda activate fraud-detection
pip install -r requirements.txt
```

## Get the data

1. Accept the competition rules at https://www.kaggle.com/c/ieee-fraud-detection
2. Download `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`
3. Place all four files in `data/raw/`

## Run the pipeline

```
python -m src.data.load_data      # merges tables, saves to data/interim/
python -m src.models.train        # trains LightGBM, logs to MLflow, saves model
```

Check training runs with:
```
mlflow ui
```

## Serve the model

```
uvicorn src.api.main:app --reload
```

Then test it:
```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"TransactionAmt\": 150.0, \"TransactionDT\": 86400, \"card1\": 1234}"
```

## Known limitation to document, not hide

The card-level aggregate features in `build_features.py` are computed
against a reference dataset. At training time that's the training split.
At serving time in `src/api/main.py`, there's no reference set wired up yet,
this needs to point at a rolling window of recent transactions (e.g. stored
in Redis or a feature store) before this is a real production system. Writing
this limitation up honestly, the same way the leakage issue got documented
in the code review paper, is worth more in an interview than pretending it's
solved.

## Project structure

- `src/data/` — loading and merging raw tables
- `src/features/` — feature engineering, shared between training and serving
- `src/models/` — training and evaluation
- `src/api/` — FastAPI serving layer
- `notebooks/` — exploratory work only, nothing here should be load-bearing
- `tests/` — unit tests for feature functions
