"""Quick end-to-end test of the full simulation pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import itertools
import joblib
import pandas as pd
from src.data.load_data import load_processed_dataset
from src.features.feature_engineering import build_feature_matrix
from src.simulation.monte_carlo import run_fixed_group_tournament
from src.simulation.confidence import compute_all_confidence_intervals
from src.simulation.groups_2026 import GROUPS_2026, TEAMS_2026
from src.simulation.retrospective import run_retrospective_validation, format_retrospective_report

print("Cargando datos y modelo...")
hist_df = load_processed_dataset()
features_df, feature_cols = build_feature_matrix(hist_df, include_elo=True, apply_decay=True, form_window=5)

home_latest = features_df.sort_values("date").groupby("home_team", as_index=False).tail(1).set_index("home_team")
away_latest = features_df.sort_values("date").groupby("away_team", as_index=False).tail(1).set_index("away_team")

pairs = [(h, a) for h, a in itertools.product(TEAMS_2026, TEAMS_2026) if h != a]
proba_base = pd.DataFrame(pairs, columns=["home_team", "away_team"])
fixture_features = proba_base.copy()
fixture_features = fixture_features.join(home_latest.add_suffix("_home"), on="home_team").join(away_latest.add_suffix("_away"), on="away_team")

fixture_features["rank_diff"] = fixture_features["rank_home_home"] - fixture_features["rank_away_away"]
fixture_features["elo_diff"] = fixture_features["elo_home_home"] - fixture_features["elo_away_away"]
fixture_features["market_value_diff"] = fixture_features["market_value_diff_home"].fillna(0) - fixture_features["market_value_diff_away"].fillna(0)
fixture_features["avg_age_diff"] = fixture_features["avg_age_diff_home"].fillna(0) - fixture_features["avg_age_diff_away"].fillna(0)
fixture_features["squad_size_diff"] = fixture_features["squad_size_diff_home"].fillna(0) - fixture_features["squad_size_diff_away"].fillna(0)
fixture_features["top5_players_diff"] = fixture_features["top5_players_diff_home"].fillna(0) - fixture_features["top5_players_diff_away"].fillna(0)
fixture_features["home_advantage"] = 0
fixture_features["form_win_rate_diff"] = fixture_features["form_win_rate_diff_home"].fillna(0) - fixture_features["form_win_rate_diff_away"].fillna(0)
fixture_features["form_goal_diff_diff"] = fixture_features["form_goal_diff_diff_home"].fillna(0) - fixture_features["form_goal_diff_diff_away"].fillna(0)
fixture_features["form_gf_avg_diff"] = fixture_features["form_gf_avg_diff_home"].fillna(0) - fixture_features["form_gf_avg_diff_away"].fillna(0)
fixture_features["form_ga_avg_diff"] = fixture_features["form_ga_avg_diff_home"].fillna(0) - fixture_features["form_ga_avg_diff_away"].fillna(0)

X_fixture = fixture_features[feature_cols]
model = joblib.load("models/trained/xgb.pkl")
proba = model.predict_proba(X_fixture)
proba_df = proba_base.copy()
proba_df["proba_0"] = proba[:, 0].astype(float)
proba_df["proba_1"] = proba[:, 1].astype(float)
proba_df["proba_2"] = proba[:, 2].astype(float)

print(f"Matriz de probabilidades: {proba_df.shape[0]} partidos")

# --- Test 1: Simulacion con grupos fijos ---
print("\nEjecutando simulacion con 1000 sims (test rapido)...")
results, winners, adv = run_fixed_group_tournament(GROUPS_2026, proba_df, n_simulations=1000, random_seed=42)

# --- Test 2: Intervalos de confianza ---
print("Calculando intervalos de confianza...")
ci_df = compute_all_confidence_intervals(winners, n_simulations=1000, confidence=0.95, n_bootstrap=1000, random_seed=42)

print("\n" + "=" * 80)
print("  TOP 10 CON INTERVALOS DE CONFIANZA (test 1K sims)")
print("=" * 80)
for _, row in ci_df.head(10).iterrows():
    pct = row["probability"] * 100
    wl = row["ci_lower_wilson"] * 100
    wu = row["ci_upper_wilson"] * 100
    bl = row["ci_lower_bootstrap"] * 100
    bu = row["ci_upper_bootstrap"] * 100
    print(f"  {row['team']:<20s} {pct:>6.2f}  Wilson:[{wl:.2f}, {wu:.2f}]  Boot:[{bl:.2f}, {bu:.2f}]")

# --- Test 3: Clasificados por grupo ---
print("\n" + "=" * 60)
print("  CLASIFICADOS POR GRUPO (top 2)")
print("=" * 60)
for g in sorted(GROUPS_2026.keys()):
    g_teams = [(t, adv[t]) for t in GROUPS_2026[g]]
    g_teams.sort(key=lambda x: x[1]["qualified"], reverse=True)
    top2 = g_teams[:2]
    p1 = top2[0][1]["qualified"] / 10
    p2 = top2[1][1]["qualified"] / 10
    print(f"  Grupo {g}: {top2[0][0]} ({p1:.1f}), {top2[1][0]} ({p2:.1f})")

# --- Test 4: Validacion retrospectiva (pocas sims) ---
print("\nEjecutando validacion retrospectiva (500 sims)...")
retro = run_retrospective_validation(proba_df, n_simulations=500, random_seed=42)
report = format_retrospective_report(retro)
print(report)

print("\n*** TODO OK - Pipeline completo verificado ***")
