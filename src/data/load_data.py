"""
Load and merge the IEEE-CIS transaction and identity tables.

Usage:
    python -m src.data.load_data
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"


def load_raw(split: str = "train") -> pd.DataFrame:
    """
    Load and merge transaction + identity tables for a given split.

    split: "train" or "test"
    """
    trans_path = RAW_DIR / f"{split}_transaction.csv"
    id_path = RAW_DIR / f"{split}_identity.csv"

    if not trans_path.exists():
        raise FileNotFoundError(
            f"{trans_path} not found. Download the IEEE-CIS dataset from Kaggle "
            f"and unzip it into data/raw/ first."
        )

    transactions = pd.read_csv(trans_path)

    # identity table doesn't cover every transaction, left join
    if id_path.exists():
        identity = pd.read_csv(id_path)
        df = transactions.merge(identity, on="TransactionID", how="left")
    else:
        df = transactions

    return df


def save_interim(df: pd.DataFrame, name: str) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    df = load_raw("train")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Fraud rate: {df['isFraud'].mean():.4f}")
    save_interim(df, "train_merged")
