from typing import Dict

import pandas as pd

from src.utils.config import DATE_COL, TARGET_COL

REQUIRED_COLUMNS = [
    DATE_COL,
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    TARGET_COL,
]


def validate_dataset(df: pd.DataFrame) -> Dict[str, int]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[DATE_COL].isna().any():
        raise ValueError("Found missing values in date column")

    if df[TARGET_COL].isna().any():
        raise ValueError("Found missing values in target column")

    return {"rows": len(df), "cols": len(df.columns)}
