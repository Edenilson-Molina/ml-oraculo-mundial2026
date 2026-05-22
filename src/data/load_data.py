import pandas as pd

from src.utils.config import DATE_COL, FINAL_DATASET


def load_processed_dataset(path: str = str(FINAL_DATASET)) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df
