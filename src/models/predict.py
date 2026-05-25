import joblib
import pandas as pd

from src.features.feature_engineering import build_feature_matrix
from src.utils.config import CLASS_LABELS


def predict_probabilities(model_path: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(model_path)
    df_features, feature_cols = build_feature_matrix(matches_df)
    X = df_features[feature_cols]

    proba = model.predict_proba(X)
    proba_df = pd.DataFrame(proba, columns=[f"proba_{label}" for label in CLASS_LABELS])
    return pd.concat([df_features.reset_index(drop=True), proba_df], axis=1)
