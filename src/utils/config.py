from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FINAL_DATASET = DATA_PROCESSED / "final_dataset.csv"

REPORTS_METRICS = PROJECT_ROOT / "reports" / "metrics"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
MODELS_TRAINED = PROJECT_ROOT / "models" / "trained"

DATE_COL = "date"
TARGET_COL = "result"
ID_COLS = ["date", "home_team", "away_team"]
LEAKAGE_COLS = ["home_score", "away_score"]
CLASS_LABELS = [0, 1, 2]

RANDOM_SEED = 42
