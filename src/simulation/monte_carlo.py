from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

GROUP_MATCHES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _match_key(home_team: str, away_team: str) -> Tuple[str, str]:
    return home_team, away_team


def _build_proba_lookup(proba_df: pd.DataFrame) -> Dict[Tuple[str, str], np.ndarray]:
    lookup: Dict[Tuple[str, str], np.ndarray] = {}
    for _, row in proba_df.iterrows():
        lookup[_match_key(row["home_team"], row["away_team"])] = np.asarray(
            row[["proba_0", "proba_1", "proba_2"]], dtype=float
        )
    return lookup


def _get_probabilities(
    lookup: Dict[Tuple[str, str], np.ndarray],
    home_team: str,
    away_team: str,
) -> np.ndarray:
    direct = lookup.get(_match_key(home_team, away_team))
    if direct is not None:
        return np.asarray(direct, dtype=float)

    reverse = lookup.get(_match_key(away_team, home_team))
    if reverse is not None:
        return np.asarray([reverse[2], reverse[1], reverse[0]], dtype=float)

    raise ValueError(f"Missing probabilities for {home_team} vs {away_team}")


def simulate_match_outcome(probabilities: np.ndarray, rng: np.random.Generator) -> Tuple[int, int, int]:
    probs = np.asarray(probabilities, dtype=float)
    total = probs.sum()
    if total <= 0:
        probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        probs = probs / total

    outcome = int(rng.choice([0, 1, 2], p=probs))
    if outcome == 2:
        return outcome, 1, 0
    if outcome == 0:
        return outcome, 0, 1
    return outcome, 0, 0


def build_groups(teams: List[str], rng: np.random.Generator) -> Dict[str, List[str]]:
    if len(teams) != 48:
        raise ValueError("Expected 48 teams for 12 groups of 4")

    shuffled = teams[:]
    rng.shuffle(shuffled)
    group_labels = [chr(ord("A") + i) for i in range(12)]
    return {
        label: shuffled[i * 4 : (i + 1) * 4] for i, label in enumerate(group_labels)
    }


def group_fixtures(groups: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for group, teams in groups.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                rows.append(
                    {
                        "group": group,
                        "home_team": teams[i],
                        "away_team": teams[j],
                    }
                )
    return pd.DataFrame(rows)


def _init_group_table(teams: Iterable[str]) -> Dict[str, Dict[str, int]]:
    return {
        team: {"points": 0, "gf": 0, "ga": 0, "gd": 0} for team in teams
    }


def _update_table(table: Dict[str, Dict[str, int]], home: str, away: str, hg: int, ag: int) -> None:
    table[home]["gf"] += hg
    table[home]["ga"] += ag
    table[away]["gf"] += ag
    table[away]["ga"] += hg
    table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
    table[away]["gd"] = table[away]["gf"] - table[away]["ga"]

    if hg > ag:
        table[home]["points"] += 3
    elif ag > hg:
        table[away]["points"] += 3
    else:
        table[home]["points"] += 1
        table[away]["points"] += 1


def _rank_group(table: Dict[str, Dict[str, int]]) -> List[str]:
    return sorted(
        table.keys(),
        key=lambda team: (
            table[team]["points"],
            table[team]["gd"],
            table[team]["gf"],
        ),
        reverse=True,
    )


def simulate_group_stage(
    groups: Dict[str, List[str]],
    lookup: Dict[Tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> Tuple[Dict[str, List[str]], List[Tuple[str, int, int, int]]]:
    group_rankings: Dict[str, List[str]] = {}
    third_place_scores: List[Tuple[str, int, int, int]] = []

    for group, teams in groups.items():
        points = np.zeros(4, dtype=int)
        gf = np.zeros(4, dtype=int)
        ga = np.zeros(4, dtype=int)

        for i, j in GROUP_MATCHES:
            home = teams[i]
            away = teams[j]
            probs = _get_probabilities(lookup, home, away)
            outcome, hg, ag = simulate_match_outcome(probs, rng)

            gf[i] += hg
            ga[i] += ag
            gf[j] += ag
            ga[j] += hg

            if outcome == 2:
                points[i] += 3
            elif outcome == 0:
                points[j] += 3
            else:
                points[i] += 1
                points[j] += 1

        gd = gf - ga
        order = np.lexsort((gf, gd, points))[::-1]
        ranking = [teams[idx] for idx in order]
        group_rankings[group] = ranking

        third_idx = order[2]
        third = teams[third_idx]
        third_place_scores.append((third, int(points[third_idx]), int(gd[third_idx]), int(gf[third_idx])))

    return group_rankings, third_place_scores


def select_best_thirds(
    third_place_scores: List[Tuple[str, int, int, int]],
    n_best: int = 8,
) -> List[str]:
    third_place_scores.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return [team for team, _, _, _ in third_place_scores[:n_best]]


def build_round_of_32(group_rankings: Dict[str, List[str]], best_thirds: List[str]) -> List[Tuple[str, str]]:
    winners = {group: ranking[0] for group, ranking in group_rankings.items()}
    runners = {group: ranking[1] for group, ranking in group_rankings.items()}

    pairs = [
        (winners["A"], best_thirds[0]),
        (winners["B"], best_thirds[1]),
        (winners["C"], best_thirds[2]),
        (winners["D"], best_thirds[3]),
        (winners["E"], best_thirds[4]),
        (winners["F"], best_thirds[5]),
        (winners["G"], best_thirds[6]),
        (winners["H"], best_thirds[7]),
        (winners["I"], runners["B"]),
        (winners["J"], runners["C"]),
        (winners["K"], runners["D"]),
        (winners["L"], runners["E"]),
        (runners["A"], runners["F"]),
        (runners["G"], runners["H"]),
        (runners["I"], runners["J"]),
        (runners["K"], runners["L"]),
    ]
    return pairs


def simulate_knockout(
    pairs: List[Tuple[str, str]],
    lookup: Dict[Tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> str:
    current_round = pairs

    while len(current_round) > 1:
        next_round = []
        for home, away in current_round:
            probs = _get_probabilities(lookup, home, away)
            outcome, _, _ = simulate_match_outcome(probs, rng)
            winner = home if outcome == 2 else away if outcome == 0 else rng.choice([home, away])
            next_round.append(winner)

        current_round = [(next_round[i], next_round[i + 1]) for i in range(0, len(next_round), 2)]

    home, away = current_round[0]
    probs = _get_probabilities(lookup, home, away)
    outcome, _, _ = simulate_match_outcome(probs, rng)
    return home if outcome == 2 else away if outcome == 0 else rng.choice([home, away])


def run_monte_carlo_tournament(
    teams: List[str],
    proba_df: pd.DataFrame,
    n_simulations: int = 10000,
    random_seed: int = 42,
) -> Tuple[Dict[str, int], List[str]]:
    logger.info("Running Monte Carlo tournament with %s simulations", n_simulations)
    lookup = _build_proba_lookup(proba_df)
    rng = np.random.default_rng(random_seed)

    winners: List[str] = []
    for _ in range(n_simulations):
        groups = build_groups(teams, rng)
        group_rankings, third_place_scores = simulate_group_stage(groups, lookup, rng)
        best_thirds = select_best_thirds(third_place_scores)
        round_of_32 = build_round_of_32(group_rankings, best_thirds)
        champion = simulate_knockout(round_of_32, lookup, rng)
        winners.append(champion)

    winner_counts = pd.Series(winners).value_counts().to_dict()
    return {str(k): int(v) for k, v in winner_counts.items()}, winners


def run_fixed_group_tournament(
    groups: Dict[str, List[str]],
    proba_df: pd.DataFrame,
    n_simulations: int = 10000,
    random_seed: int = 42,
) -> Tuple[Dict[str, int], List[str], Dict[str, Dict[str, int]]]:
    """Run a Monte Carlo tournament with fixed group assignments.

    Unlike ``run_monte_carlo_tournament``, this function keeps the same
    group composition across every simulation iteration instead of
    reshuffling teams into random groups.

    Parameters
    ----------
    groups : Dict[str, List[str]]
        Pre-defined groups mapping group label to a list of four teams.
    proba_df : pd.DataFrame
        Match probability matrix with columns
        ``home_team``, ``away_team``, ``proba_0``, ``proba_1``, ``proba_2``.
    n_simulations : int, optional
        Number of tournament simulations to run, by default 10000.
    random_seed : int, optional
        Seed for the random number generator, by default 42.

    Returns
    -------
    Tuple[Dict[str, int], List[str], Dict[str, Dict[str, int]]]
        * ``winner_counts`` – mapping of team name to championship count.
        * ``winners`` – full list of champion per simulation.
        * ``advancement`` – for each team a dict with keys
          ``"group"``, ``"qualified"``, ``"first"``, ``"second"``,
          ``"third"``.
    """
    logger.info(
        "Running fixed-group Monte Carlo tournament with %s simulations",
        n_simulations,
    )
    lookup = _build_proba_lookup(proba_df)
    rng = np.random.default_rng(random_seed)

    # Initialise advancement tracking
    advancement: Dict[str, Dict[str, int]] = {}
    for group_label, team_list in groups.items():
        for team in team_list:
            advancement[team] = {
                "group": group_label,
                "qualified": 0,
                "first": 0,
                "second": 0,
                "third": 0,
            }

    winners: List[str] = []
    for _ in range(n_simulations):
        group_rankings, third_place_scores = simulate_group_stage(
            groups, lookup, rng
        )

        # Record positional finishes
        for group_label, ranking in group_rankings.items():
            if len(ranking) >= 1:
                advancement[ranking[0]]["first"] += 1
                advancement[ranking[0]]["qualified"] += 1
            if len(ranking) >= 2:
                advancement[ranking[1]]["second"] += 1
                advancement[ranking[1]]["qualified"] += 1
            if len(ranking) >= 3:
                advancement[ranking[2]]["third"] += 1

        best_thirds = select_best_thirds(third_place_scores)
        round_of_32 = build_round_of_32(group_rankings, best_thirds)
        champion = simulate_knockout(round_of_32, lookup, rng)
        winners.append(champion)

    winner_counts = pd.Series(winners).value_counts().to_dict()
    return (
        {str(k): int(v) for k, v in winner_counts.items()},
        winners,
        advancement,
    )
