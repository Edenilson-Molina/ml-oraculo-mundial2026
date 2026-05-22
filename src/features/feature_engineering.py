from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data_processing import apply_time_decay, calculate_elo
from src.utils.config import DATE_COL, LEAKAGE_COLS, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_row_id"] = np.arange(len(df))
    df = df.sort_values(DATE_COL)
    df = calculate_elo(df)
    df = df.sort_values("_row_id").drop(columns=["_row_id"])
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df


def add_recent_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    if "neutral" not in df.columns:
        df["neutral"] = False

    df["match_id"] = np.arange(len(df))

    home = df[["match_id", DATE_COL, "home_team", "home_score", "away_score"]].copy()
    home = home.rename(columns={"home_team": "team"})
    home["is_home"] = True
    home["goals_for"] = home["home_score"]
    home["goals_against"] = home["away_score"]

    away = df[["match_id", DATE_COL, "away_team", "home_score", "away_score"]].copy()
    away = away.rename(columns={"away_team": "team"})
    away["is_home"] = False
    away["goals_for"] = away["away_score"]
    away["goals_against"] = away["home_score"]

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", DATE_COL, "match_id"])

    long_df["is_win"] = (long_df["goals_for"] > long_df["goals_against"]).astype(int)
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]

    group = long_df.groupby("team", sort=False)
    long_df["form_win_rate"] = group["is_win"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long_df["form_gf_avg"] = group["goals_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long_df["form_ga_avg"] = group["goals_against"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long_df["form_goal_diff"] = group["goal_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home_form = long_df[long_df["is_home"]].copy()
    away_form = long_df[~long_df["is_home"]].copy()

    home_form = home_form[["match_id", "form_win_rate", "form_gf_avg", "form_ga_avg", "form_goal_diff"]]
    away_form = away_form[["match_id", "form_win_rate", "form_gf_avg", "form_ga_avg", "form_goal_diff"]]

    home_form = home_form.rename(
        columns={
            "form_win_rate": "form_win_rate_home",
            "form_gf_avg": "form_gf_avg_home",
            "form_ga_avg": "form_ga_avg_home",
            "form_goal_diff": "form_goal_diff_home",
        }
    )
    away_form = away_form.rename(
        columns={
            "form_win_rate": "form_win_rate_away",
            "form_gf_avg": "form_gf_avg_away",
            "form_ga_avg": "form_ga_avg_away",
            "form_goal_diff": "form_goal_diff_away",
        }
    )

    df = df.merge(home_form, on="match_id", how="left")
    df = df.merge(away_form, on="match_id", how="left")

    df["form_win_rate_diff"] = df["form_win_rate_home"] - df["form_win_rate_away"]
    df["form_gf_avg_diff"] = df["form_gf_avg_home"] - df["form_gf_avg_away"]
    df["form_ga_avg_diff"] = df["form_ga_avg_home"] - df["form_ga_avg_away"]
    df["form_goal_diff_diff"] = df["form_goal_diff_home"] - df["form_goal_diff_away"]

    df = df.drop(columns=["match_id"])
    return df


def add_basic_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "rank_diff" not in df.columns and {"rank_home", "rank_away"}.issubset(df.columns):
        df["rank_diff"] = df["rank_home"] - df["rank_away"]

    if "market_value_diff" not in df.columns and {"market_value_eur_home", "market_value_eur_away"}.issubset(df.columns):
        df["market_value_diff"] = df["market_value_eur_home"] - df["market_value_eur_away"]

    if "avg_age_diff" not in df.columns and {"avg_age_home", "avg_age_away"}.issubset(df.columns):
        df["avg_age_diff"] = df["avg_age_home"] - df["avg_age_away"]

    if "squad_size_diff" not in df.columns and {"squad_size_home", "squad_size_away"}.issubset(df.columns):
        df["squad_size_diff"] = df["squad_size_home"] - df["squad_size_away"]

    if "top5_players_diff" not in df.columns and {"top5_players_home", "top5_players_away"}.issubset(df.columns):
        df["top5_players_diff"] = df["top5_players_home"] - df["top5_players_away"]

    if "neutral" in df.columns:
        df["home_advantage"] = (~df["neutral"].astype(bool)).astype(int)

    return df


def add_features(
    df: pd.DataFrame,
    include_elo: bool = True,
    apply_decay: bool = True,
    form_window: int = 5,
) -> pd.DataFrame:
    df = df.copy()

    if include_elo:
        logger.info("Adding ELO features")
        df = add_elo_features(df)

    logger.info("Adding recent form features")
    df = add_recent_form_features(df, window=form_window)

    logger.info("Adding basic diff features")
    df = add_basic_diff_features(df)

    if apply_decay:
        logger.info("Applying time decay")
        df = apply_time_decay(df)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    base_features = [
        "rank_diff",
        "elo_diff",
        "market_value_diff",
        "avg_age_diff",
        "squad_size_diff",
        "top5_players_diff",
        "home_advantage",
        "form_win_rate_diff",
        "form_goal_diff_diff",
        "form_gf_avg_diff",
        "form_ga_avg_diff",
    ]

    feature_cols = [col for col in base_features if col in df.columns]
    leakage = set(LEAKAGE_COLS + [TARGET_COL])
    feature_cols = [col for col in feature_cols if col not in leakage]

    if not feature_cols:
        raise ValueError("No valid feature columns found after engineering")

    return feature_cols


def build_feature_matrix(
    df: pd.DataFrame,
    include_elo: bool = True,
    apply_decay: bool = True,
    form_window: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    df_features = add_features(
        df,
        include_elo=include_elo,
        apply_decay=apply_decay,
        form_window=form_window,
    )
    feature_cols = get_feature_columns(df_features)
    return df_features, feature_cols
