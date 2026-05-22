import argparse
from typing import Dict

import joblib
import numpy as np

from src.data.load_data import load_processed_dataset
from src.evaluation.metrics import evaluate_predictions
from src.features.feature_engineering import build_feature_matrix
from src.utils.config import CLASS_LABELS, DATE_COL, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model_path: str, form_window: int = 5) -> Dict[str, float]:
    model = joblib.load(model_path)

    df = load_processed_dataset()
    df_features, feature_cols = build_feature_matrix(
        df,
        include_elo=True,
        apply_decay=True,
        form_window=form_window,
    )

    df_features = df_features.sort_values(DATE_COL).reset_index(drop=True)
    X = df_features[feature_cols]
    y = df_features[TARGET_COL].to_numpy()

    y_proba = model.predict_proba(X)
    metrics = evaluate_predictions(y, y_proba, CLASS_LABELS)

    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--form-window", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(args.model_path, form_window=args.form_window)


if __name__ == "__main__":
    main()
