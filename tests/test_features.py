import pandas as pd
import numpy as np
from src.features.build_features import add_time_features, add_amount_features

def make_dummy_df():
    return pd.DataFrame({
        "TransactionDT": [0, 3600, 90000],
        "TransactionAmt": [100.0, 49.99, 200.5],
    })


def test_add_time_features():
    df = add_time_features(make_dummy_df())
    assert "hour" in df.columns
    assert "day" in df.columns
    assert df["hour"].iloc[1] == 1


def test_add_amount_features():
    df = add_amount_features(make_dummy_df())
    assert "TransactionAmt_log" in df.columns
    assert np.isclose(df["TransactionAmt_log"].iloc[0], np.log1p(100.0))
