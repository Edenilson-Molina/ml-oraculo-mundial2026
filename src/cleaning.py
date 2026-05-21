"""
Pipeline de limpieza y feature engineering.

Genera un dataset final con:
- Partidos + rankings FIFA (merge temporal)
- Calidad de plantilla (Transfermarkt)
- Features derivadas basicas

Uso:
    python src/cleaning.py
"""

import os
import pandas as pd

from data_processing import load_and_clean_matches, apply_time_decay

# ============================================
# RUTAS
# ============================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

MATCHES_PATH = os.path.join(DATA_RAW, "01_historical_matches.csv")
RANKING_PATH = os.path.join(DATA_RAW, "02_fifa_rankings.csv")
SQUAD_PATH = os.path.join(DATA_RAW, "03_squad_quality.csv")

OUTPUT_PATH = os.path.join(DATA_PROCESSED, "final_dataset.csv")

os.makedirs(DATA_PROCESSED, exist_ok=True)

# ============================================
# NORMALIZACION DE NOMBRES
# ============================================
TEAM_NAME_MAP = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Iran IR": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "Congo DR",
    "China PR": "China",
    "Korea DPR": "North Korea",
    "Russia": "Russia",
    "Turkey": "Turkey",
    "Czech Republic": "Czechia",
}


def normalize_team_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


# ============================================
# PIPELINE PRINCIPAL
# ============================================

def build_dataset(apply_decay: bool = True) -> pd.DataFrame:
    # 1) Base: partidos + ranking FIFA
    df = load_and_clean_matches(MATCHES_PATH, RANKING_PATH)

    # 2) Cargar calidad de plantilla
    squad = pd.read_csv(SQUAD_PATH)

    # 3) Normalizar nombres para el merge
    df["home_team_std"] = df["home_team"].apply(normalize_team_name)
    df["away_team_std"] = df["away_team"].apply(normalize_team_name)
    squad["team_name"] = squad["team_name"].apply(normalize_team_name)

    # 4) Merge de calidad de plantilla (home)
    df = df.merge(
        squad[["team_name", "squad_size", "avg_age", "market_value_eur", "fifa_rank", "top5_players"]],
        left_on="home_team_std",
        right_on="team_name",
        how="left",
    ).rename(
        columns={
            "squad_size": "squad_size_home",
            "avg_age": "avg_age_home",
            "market_value_eur": "market_value_eur_home",
            "fifa_rank": "fifa_rank_home",
            "top5_players": "top5_players_home",
        }
    ).drop(columns=["team_name"])

    # 5) Merge de calidad de plantilla (away)
    df = df.merge(
        squad[["team_name", "squad_size", "avg_age", "market_value_eur", "fifa_rank", "top5_players"]],
        left_on="away_team_std",
        right_on="team_name",
        how="left",
    ).rename(
        columns={
            "squad_size": "squad_size_away",
            "avg_age": "avg_age_away",
            "market_value_eur": "market_value_eur_away",
            "fifa_rank": "fifa_rank_away",
            "top5_players": "top5_players_away",
        }
    ).drop(columns=["team_name"])

    # 6) Features derivadas basicas
    # Nota: usamos diferencias para evitar escalas dispares en el modelo.
    df["market_value_diff"] = df["market_value_eur_home"] - df["market_value_eur_away"]
    df["avg_age_diff"] = df["avg_age_home"] - df["avg_age_away"]
    df["squad_size_diff"] = df["squad_size_home"] - df["squad_size_away"]
    df["top5_players_diff"] = df["top5_players_home"] - df["top5_players_away"]
    df["rank_diff"] = df["rank_home"] - df["rank_away"]

    # 7) Time decay opcional
    if apply_decay:
        df = apply_time_decay(df)

    # 8) Limpieza final de columnas auxiliares
    df = df.drop(columns=["home_team_std", "away_team_std"])

    return df


def main():
    print("=" * 60)
    print("CLEANING & FEATURE ENGINEERING")
    print("=" * 60)

    df_final = build_dataset(apply_decay=True)
    df_final.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset final guardado en: {OUTPUT_PATH}")
    print(f"Filas: {len(df_final)} | Columnas: {len(df_final.columns)}")

    missing = df_final.isna().mean().sort_values(ascending=False).head(10)
    print("Top 10 columnas con NA:")
    for col, rate in missing.items():
        print(f"   - {col}: {rate:.1%}")


if __name__ == "__main__":
    main()
