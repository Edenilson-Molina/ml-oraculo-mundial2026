"""Retrospective validation – simulate past World Cups (2018 & 2022).

Uses the trained model's match-probability matrix to run Monte-Carlo
simulations of historic 32-team tournaments and compares the predicted
winners against actual results.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.simulation.monte_carlo import (
    _build_proba_lookup,
    _get_probabilities,
    simulate_match_outcome,
    simulate_group_stage,
    simulate_knockout,
    GROUP_MATCHES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# World Cup 2018 (Russia) – 32 teams, 8 groups of 4
# ---------------------------------------------------------------------------
WC2018_GROUPS: Dict[str, List[str]] = {
    "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
    "B": ["Portugal", "Spain", "Morocco", "Iran"],
    "C": ["France", "Australia", "Peru", "Denmark"],
    "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
    "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
    "F": ["Germany", "Mexico", "Sweden", "South Korea"],
    "G": ["Belgium", "Panama", "Tunisia", "England"],
    "H": ["Poland", "Senegal", "Colombia", "Japan"],
}

WC2018_ACTUAL_WINNER: str = "France"
WC2018_ACTUAL_TOP4: List[str] = ["France", "Croatia", "Belgium", "England"]

# ---------------------------------------------------------------------------
# World Cup 2022 (Qatar) – 32 teams, 8 groups of 4
# ---------------------------------------------------------------------------
WC2022_GROUPS: Dict[str, List[str]] = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

WC2022_ACTUAL_WINNER: str = "Argentina"
WC2022_ACTUAL_TOP4: List[str] = ["Argentina", "France", "Croatia", "Morocco"]

# ---------------------------------------------------------------------------
# 32-team bracket: Round-of-16 matchups by group position
# ---------------------------------------------------------------------------
_R16_BRACKET_32: List[Tuple[str, str]] = [
    # (group_winner, group_runner-up)
    ("A", "B"),  # 1A vs 2B
    ("C", "D"),  # 1C vs 2D
    ("E", "F"),  # 1E vs 2F
    ("G", "H"),  # 1G vs 2H
    ("B", "A"),  # 1B vs 2A
    ("D", "C"),  # 1D vs 2C
    ("F", "E"),  # 1F vs 2E
    ("H", "G"),  # 1H vs 2G
]


def _build_round_of_16(
    group_rankings: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """Build Round-of-16 pairings for a 32-team World Cup.

    The bracket follows the standard FIFA format:
    1A v 2B, 1C v 2D, 1E v 2F, 1G v 2H,
    1B v 2A, 1D v 2C, 1F v 2E, 1H v 2G.

    Parameters
    ----------
    group_rankings:
        Mapping *group_label → ordered list of teams* (best first).

    Returns
    -------
    List of 8 (home, away) tuples for the Round of 16.
    """
    pairs: List[Tuple[str, str]] = []
    for winner_group, runner_group in _R16_BRACKET_32:
        home = group_rankings[winner_group][0]   # group winner
        away = group_rankings[runner_group][1]    # group runner-up
        pairs.append((home, away))
    return pairs


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_past_worldcup_32(
    groups: Dict[str, List[str]],
    proba_df: pd.DataFrame,
    n_simulations: int = 10_000,
    random_seed: int = 42,
) -> Tuple[Dict[str, int], List[str]]:
    """Monte-Carlo simulation of a 32-team World Cup.

    Reuses :func:`simulate_group_stage` and :func:`simulate_knockout`
    from *monte_carlo.py*, but applies the 32-team bracket where the
    top 2 from each group advance directly to the Round of 16 (no
    best-third-place rule).

    Parameters
    ----------
    groups:
        Fixed group draw – 8 groups of 4 teams each.
    proba_df:
        DataFrame with columns ``home_team``, ``away_team``,
        ``proba_0``, ``proba_1``, ``proba_2``.
    n_simulations:
        Number of tournament simulations to run.
    random_seed:
        Seed for reproducibility.

    Returns
    -------
    winner_counts:
        ``{team_name: n_wins}`` sorted by frequency (descending).
    winners_list:
        Full list of winners (length == *n_simulations*).
    """
    logger.info(
        "Simulating 32-team World Cup (%d simulations, seed=%d)",
        n_simulations,
        random_seed,
    )
    lookup = _build_proba_lookup(proba_df)
    rng = np.random.default_rng(random_seed)

    winners: List[str] = []
    for _ in range(n_simulations):
        # Group stage – reuse existing function (works for 4-team groups)
        group_rankings, _third_place_scores = simulate_group_stage(
            groups, lookup, rng
        )

        # Build Round of 16 (top 2 per group, no best thirds)
        r16_pairs = _build_round_of_16(group_rankings)

        # Knockout rounds (R16 → QF → SF → Final)
        champion = simulate_knockout(r16_pairs, lookup, rng)
        winners.append(champion)

    winner_counts: Dict[str, int] = (
        pd.Series(winners).value_counts().to_dict()
    )
    winner_counts = {str(k): int(v) for k, v in winner_counts.items()}

    logger.info("Simulation complete. Top predicted winner: %s", next(iter(winner_counts)))
    return winner_counts, winners


# ---------------------------------------------------------------------------
# Retrospective validation
# ---------------------------------------------------------------------------

def run_retrospective_validation(
    proba_df: pd.DataFrame,
    n_simulations: int = 10_000,
    random_seed: int = 42,
) -> Dict[str, Dict]:
    """Run retrospective validation for WC 2018 and WC 2022.

    Simulates both past tournaments and compares predictions with
    actual outcomes.

    Parameters
    ----------
    proba_df:
        Full match-probability matrix (all pairwise matchups).
    n_simulations:
        Number of tournament simulations per edition.
    random_seed:
        Base random seed; WC 2022 uses ``random_seed + 1`` for
        independence.

    Returns
    -------
    Dictionary keyed by ``"2018"`` and ``"2022"``, each containing:

    - ``predictions`` – ``{team: win_count}``
    - ``actual_winner`` – the real-life champion
    - ``actual_top4`` – the real-life top-4 finishers
    - ``winner_predicted_rank`` – rank of the actual winner in
      the model's predicted standings (1 = most predicted)
    - ``winner_in_top5`` – whether the actual winner appears in
      the model's top 5
    - ``top4_predicted_ranks`` – list of predicted ranks for
      each real-life top-4 team
    """
    logger.info("Starting retrospective validation for WC 2018 & 2022")

    results: Dict[str, Dict] = {}

    editions = [
        ("2018", WC2018_GROUPS, WC2018_ACTUAL_WINNER, WC2018_ACTUAL_TOP4, random_seed),
        ("2022", WC2022_GROUPS, WC2022_ACTUAL_WINNER, WC2022_ACTUAL_TOP4, random_seed + 1),
    ]

    for year, groups, actual_winner, actual_top4, seed in editions:
        logger.info("Running retrospective simulation for WC %s", year)

        winner_counts, _ = simulate_past_worldcup_32(
            groups=groups,
            proba_df=proba_df,
            n_simulations=n_simulations,
            random_seed=seed,
        )

        # Rank teams by predicted win frequency (1 = most predicted)
        sorted_teams = sorted(
            winner_counts.items(), key=lambda x: x[1], reverse=True
        )
        team_rank = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

        # Actual winner rank (fallback to total participant count + 1 if
        # the model never predicted that team as champion)
        total_teams = sum(len(g) for g in groups.values())
        winner_rank = team_rank.get(actual_winner, total_teams + 1)

        # Ranks for each top-4 finisher
        top4_ranks = [
            team_rank.get(team, total_teams + 1) for team in actual_top4
        ]

        top10 = sorted_teams[:10]

        results[year] = {
            "predictions": winner_counts,
            "top10": top10,
            "actual_winner": actual_winner,
            "actual_top4": actual_top4,
            "winner_predicted_rank": winner_rank,
            "winner_in_top5": winner_rank <= 5,
            "top4_predicted_ranks": top4_ranks,
        }

        logger.info(
            "WC %s – actual winner '%s' ranked #%d in predictions (in top 5: %s)",
            year,
            actual_winner,
            winner_rank,
            winner_rank <= 5,
        )

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_retrospective_report(validation_results: Dict[str, Dict]) -> str:
    """Format retrospective validation results as a human-readable report.

    Parameters
    ----------
    validation_results:
        Output of :func:`run_retrospective_validation`.

    Returns
    -------
    Multi-line string ready for printing or saving to a file.
    """
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("  RETROSPECTIVE VALIDATION – Past World Cup Simulations")
    lines.append("=" * 70)
    lines.append("")

    for year in sorted(validation_results.keys()):
        data = validation_results[year]
        lines.append("-" * 70)
        lines.append(f"  World Cup {year}")
        lines.append("-" * 70)

        # Top 10 predicted winners
        lines.append("")
        lines.append("  Top 10 Predicted Winners:")
        for rank, (team, count) in enumerate(data["top10"], start=1):
            total_sims = sum(data["predictions"].values())
            pct = count / total_sims * 100
            marker = " ◄ ACTUAL WINNER" if team == data["actual_winner"] else ""
            lines.append(f"    {rank:>2}. {team:<25s} {count:>5d} wins ({pct:5.1f}%){marker}")

        # Actual results comparison
        lines.append("")
        lines.append(f"  Actual Winner:       {data['actual_winner']}")
        lines.append(f"  Predicted Rank:      #{data['winner_predicted_rank']}")
        lines.append(f"  In Top 5:            {'Yes ✓' if data['winner_in_top5'] else 'No ✗'}")

        lines.append("")
        lines.append("  Actual Top 4 vs Predicted Ranks:")
        for team, rank in zip(data["actual_top4"], data["top4_predicted_ranks"]):
            lines.append(f"    {team:<25s}  → Predicted rank #{rank}")

        lines.append("")

    # Summary
    lines.append("=" * 70)
    lines.append("  Summary")
    lines.append("=" * 70)
    winners_in_top5 = sum(
        1 for d in validation_results.values() if d["winner_in_top5"]
    )
    total = len(validation_results)
    lines.append(f"  Winners in top 5:    {winners_in_top5}/{total}")

    avg_rank = np.mean(
        [d["winner_predicted_rank"] for d in validation_results.values()]
    )
    lines.append(f"  Avg winner rank:     {avg_rank:.1f}")

    all_top4_ranks = []
    for d in validation_results.values():
        all_top4_ranks.extend(d["top4_predicted_ranks"])
    avg_top4 = np.mean(all_top4_ranks)
    lines.append(f"  Avg top-4 rank:      {avg_top4:.1f}")

    lines.append("=" * 70)
    lines.append("")
    return "\n".join(lines)
