"""
Feature engineering for the IEEE-CIS fraud dataset.

Keep every function here pure (no reading/writing files) so it's testable
and so the exact same pipeline can be reused at inference time in the API.
"""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """TransactionDT is seconds from an arbitrary reference point, not a real timestamp."""
    df = df.copy()
    df["day"] = df["TransactionDT"] // (24 * 60 * 60)
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform amount and flag round-number amounts, which correlate with fraud."""
    df = df.copy()
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = ((df["TransactionAmt"] - df["TransactionAmt"].astype(int)) * 1000).astype(int)
    return df


def add_card_aggregates(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate stats per card, computed on ref_df (should be training data / past data only)
    and mapped onto df. This split matters: computing aggregates on the full dataset
    including future rows is a leakage bug, same category as the Task 2 leakage you
    caught in the code review paper.
    """
    df = df.copy()
    ref = ref_df if ref_df is not None else df

    card_stats = ref.groupby("card1")["TransactionAmt"].agg(["mean", "std", "count"])
    card_stats.columns = ["card1_amt_mean", "card1_amt_std", "card1_txn_count"]

    df = df.merge(card_stats, on="card1", how="left")
    df["amt_vs_card_mean"] = df["TransactionAmt"] / (df["card1_amt_mean"] + 1e-5)
    return df


def add_missing_flags(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    """Missingness on identity columns can itself be a fraud signal, don't just impute it away."""
    df = df.copy()
    df["identity_missing_count"] = df[id_cols].isna().sum(axis=1)
    return df


def build_feature_pipeline(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
    """Run the full feature pipeline in order. This is what both training and the API should call."""
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_card_aggregates(df, ref_df=ref_df)
    id_cols = [c for c in df.columns if c.startswith("id_")]
    if id_cols:
        df = add_missing_flags(df, id_cols)
    return df
