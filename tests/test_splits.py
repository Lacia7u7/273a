import numpy as np
import pandas as pd

from src.data.splits import assert_no_group_leakage, make_splits
from src.utils.config import SplitStrategy


def test_group_kfold_no_leakage():
    df = pd.DataFrame({"patient": [1, 1, 2, 3, 4, 4], "target": [0, 1, 0, 1, 0, 1]})
    strategy = SplitStrategy(strategy="group_k_fold", group_by="patient", n_splits=3, seed=42, stratify_by_target=False)
    splits = make_splits(df, strategy, "patient", df["target"].values)
    assert len(splits) == 3
    assert_no_group_leakage(df.assign(patient=df["patient"]), splits, "patient")
