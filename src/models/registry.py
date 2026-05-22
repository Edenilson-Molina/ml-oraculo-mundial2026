import json
from pathlib import Path
from typing import Dict

import joblib

from src.utils.config import MODELS_TRAINED, REPORTS_METRICS


def save_model(model, model_name: str) -> Path:
    MODELS_TRAINED.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_TRAINED / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    return model_path


def save_metrics(metrics: Dict, model_name: str) -> Path:
    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_METRICS / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics_path
