from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import FEATURE_NAMES, ID_COLUMN, TARGET_COLUMN


def load_credit_card_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(data_path)
    expected_columns = set(FEATURE_NAMES + [TARGET_COLUMN])
    missing = sorted(expected_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    x = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    x = x[FEATURE_NAMES]
    y = frame[TARGET_COLUMN].astype(int)
    return x, y
