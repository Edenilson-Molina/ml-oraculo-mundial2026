import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.load_data import load_processed_dataset
from src.data.validation import validate_dataset
from src.evaluation.metrics import evaluate_predictions
from src.features.feature_engineering import build_feature_matrix
from src.models.registry import save_metrics, save_model
from src.utils.config import CLASS_LABELS, DATE_COL, RANDOM_SEED, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def temporal_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_idx = df_sorted.index[:split_idx].to_numpy()
    test_idx = df_sorted.index[split_idx:].to_numpy()
    return train_idx, test_idx


def build_models() -> Dict[str, object]:
    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )

    return {"logreg": logreg, "xgb": xgb}


def _build_fit_params(model, sample_weight: Optional[np.ndarray]) -> Dict:
    if sample_weight is None:
        return {}

    if hasattr(model, "named_steps"):
        return {"clf__sample_weight": sample_weight}

    if hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        return {"clf__sample_weight": sample_weight}

    return {"sample_weight": sample_weight}


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray],
    cv: int = 5,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in skf.split(X, y):
        model_clone = clone(model)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        weights_train = sample_weight[train_idx] if sample_weight is not None else None
        fit_params = _build_fit_params(model_clone, weights_train)
        model_clone.fit(X_train, y_train, **fit_params)

        y_proba = model_clone.predict_proba(X_val)
        metrics = evaluate_predictions(y_val.to_numpy(), y_proba, CLASS_LABELS)
        fold_metrics.append(metrics)

    mean_metrics = {
        key: float(np.mean([m[key] for m in fold_metrics])) for key in fold_metrics[0]
    }
    return mean_metrics


def _get_param_search_space() -> Dict[str, Dict]:
    return {
        "logreg": {
            "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "clf__solver": ["lbfgs", "saga"],
        },
        "xgb": {
            "n_estimators": [200, 400, 600],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        },
    }


def _tune_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray],
    n_iter: int = 10,
) -> object:
    param_space = _get_param_search_space().get(name)
    if not param_space:
        return model

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )

    fit_params = _build_fit_params(search, sample_weight)
    search.fit(X, y, **fit_params)
    logger.info("Best params for %s: %s", name, search.best_params_)
    return search.best_estimator_


def train_models(
    test_size: float = 0.2,
    form_window: int = 5,
    use_search: bool = False,
    n_iter: int = 10,
) -> Dict[str, Dict]:
    df = load_processed_dataset()
    validate_dataset(df)

    df_features, feature_cols = build_feature_matrix(
        df,
        include_elo=True,
        apply_decay=True,
        form_window=form_window,
    )

    df_features = df_features.sort_values(DATE_COL).reset_index(drop=True)

    X = df_features[feature_cols]
    y = df_features[TARGET_COL]
    weights = df_features["weight"].to_numpy() if "weight" in df_features.columns else None

    train_idx, test_idx = temporal_split(df_features, test_size=test_size)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    w_train = weights[train_idx] if weights is not None else None
    w_test = weights[test_idx] if weights is not None else None

    results: Dict[str, Dict] = {}

    for name, model in build_models().items():
        logger.info("Training model: %s", name)
        if use_search:
            logger.info("Running hyperparameter search for: %s", name)
            model = _tune_model(name, model, X_train, y_train, w_train, n_iter=n_iter)

        cv_metrics = cross_validate_model(model, X_train, y_train, w_train)

        fit_params = _build_fit_params(model, w_train)
        model.fit(X_train, y_train, **fit_params)

        y_proba_test = model.predict_proba(X_test)
        test_metrics = evaluate_predictions(y_test.to_numpy(), y_proba_test, CLASS_LABELS)

        metrics = {
            "cv": cv_metrics,
            "test": test_metrics,
            "features": feature_cols,
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
        }

        model_path = save_model(model, name)
        metrics_path = save_metrics(metrics, name)

        results[name] = {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        }

        logger.info("Saved model to %s", model_path)
        logger.info("Saved metrics to %s", metrics_path)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--form-window", type=int, default=5)
    parser.add_argument("--search", action="store_true", help="Enable hyperparameter search")
    parser.add_argument("--n-iter", type=int, default=10, help="Iterations for search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_models(
        test_size=args.test_size,
        form_window=args.form_window,
        use_search=args.search,
        n_iter=args.n_iter,
    )


if __name__ == "__main__":
    main()
