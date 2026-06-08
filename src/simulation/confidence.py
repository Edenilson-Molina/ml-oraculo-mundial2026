"""Confidence interval estimation for Monte Carlo tournament simulations.

Provides two complementary approaches for quantifying the uncertainty
of championship probabilities obtained from Monte Carlo simulations:

1. **Wilson Score Interval** – an analytical binomial CI that behaves
   well even for extreme proportions and small sample sizes.
2. **Bootstrap Percentile Interval** – a non-parametric CI obtained by
   resampling the winner list with replacement.

A convenience function combines both methods into a single DataFrame.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Wilson Score Interval
# ---------------------------------------------------------------------------

def compute_wilson_ci(
    wins: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute the Wilson Score confidence interval for a binomial proportion.

    The Wilson interval is preferred over the normal (Wald) interval
    because it has better coverage properties, especially when *p* is
    near 0 or 1, or when *n* is small.

    Parameters
    ----------
    wins : int
        Number of successes (e.g. championships won by a team).
    n : int
        Total number of trials (simulations).
    confidence : float, optional
        Confidence level, by default 0.95.

    Returns
    -------
    Tuple[float, float, float]
        ``(p_hat, ci_lower, ci_upper)`` where *p_hat* is the raw
        sample proportion and *ci_lower* / *ci_upper* are the Wilson
        Score interval bounds.

    Raises
    ------
    ValueError
        If *n* is not positive or *wins* is negative / exceeds *n*.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if wins < 0 or wins > n:
        raise ValueError(
            f"wins must satisfy 0 <= wins <= n, got wins={wins}, n={n}"
        )

    p_hat: float = wins / n
    z: float = norm.ppf(1 - (1 - confidence) / 2)
    z2: float = z * z

    denominator: float = 1 + z2 / n
    center: float = (p_hat + z2 / (2 * n)) / denominator
    margin: float = (
        z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    ) / denominator

    ci_lower: float = max(0.0, center - margin)
    ci_upper: float = min(1.0, center + margin)

    logger.debug(
        "Wilson CI (wins=%d, n=%d, conf=%.2f): p_hat=%.4f, [%.4f, %.4f]",
        wins,
        n,
        confidence,
        p_hat,
        ci_lower,
        ci_upper,
    )

    return p_hat, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# 2. Bootstrap Percentile Interval
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    winners: List[str],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compute bootstrap percentile confidence intervals for each team.

    The procedure resamples the *winners* list with replacement
    ``n_bootstrap`` times.  For every resample the championship
    proportion of each team is computed, and the resulting distribution
    is summarised with the percentile CI.

    Parameters
    ----------
    winners : List[str]
        Full list of tournament winners across Monte Carlo simulations
        (one entry per simulation).
    n_bootstrap : int, optional
        Number of bootstrap resamples, by default 10 000.
    confidence : float, optional
        Confidence level, by default 0.95.
    random_seed : int, optional
        Seed for the random number generator, by default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``team``, ``probability``,
        ``ci_lower_bootstrap``, ``ci_upper_bootstrap``.
    """
    logger.info(
        "Computing bootstrap CI with %d resamples (seed=%d, conf=%.2f)",
        n_bootstrap,
        random_seed,
        confidence,
    )

    n: int = len(winners)
    if n == 0:
        logger.warning("Empty winners list – returning empty DataFrame")
        return pd.DataFrame(
            columns=["team", "probability", "ci_lower_bootstrap", "ci_upper_bootstrap"]
        )

    rng = np.random.default_rng(random_seed)
    winners_arr: np.ndarray = np.asarray(winners)
    unique_teams: List[str] = sorted(set(winners))

    # Pre-compute observed proportions
    observed_counts = pd.Series(winners).value_counts()

    # Matrix to hold bootstrap proportions: (n_bootstrap, n_teams)
    boot_proportions: np.ndarray = np.zeros(
        (n_bootstrap, len(unique_teams)), dtype=float
    )
    team_to_idx = {team: idx for idx, team in enumerate(unique_teams)}

    for b in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        sample = winners_arr[sample_indices]
        unique_vals, counts = np.unique(sample, return_counts=True)
        for team, count in zip(unique_vals, counts):
            boot_proportions[b, team_to_idx[team]] = count / n

    alpha: float = 1 - confidence
    lower_pct: float = (alpha / 2) * 100
    upper_pct: float = (1 - alpha / 2) * 100

    rows: List[dict] = []
    for team in unique_teams:
        idx = team_to_idx[team]
        col = boot_proportions[:, idx]
        ci_lo: float = float(np.percentile(col, lower_pct))
        ci_hi: float = float(np.percentile(col, upper_pct))
        prob: float = float(observed_counts.get(team, 0) / n)
        rows.append(
            {
                "team": team,
                "probability": prob,
                "ci_lower_bootstrap": ci_lo,
                "ci_upper_bootstrap": ci_hi,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Bootstrap CI computed for %d teams", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Combined Confidence Intervals
# ---------------------------------------------------------------------------

def compute_all_confidence_intervals(
    winners: List[str],
    n_simulations: int,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compute Wilson Score and Bootstrap CIs for every team.

    Merges the analytical Wilson interval with the non-parametric
    Bootstrap percentile interval into a single, sorted DataFrame.

    Parameters
    ----------
    winners : List[str]
        Full list of tournament winners across Monte Carlo simulations.
    n_simulations : int
        Total number of simulations that were run (used as *n* for the
        Wilson formula).
    confidence : float, optional
        Confidence level for both intervals, by default 0.95.
    n_bootstrap : int, optional
        Number of bootstrap resamples, by default 10 000.
    random_seed : int, optional
        Seed for the bootstrap RNG, by default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        * ``team``
        * ``championships`` – raw win count
        * ``probability`` – *p̂*
        * ``ci_lower_wilson``, ``ci_upper_wilson``
        * ``ci_lower_bootstrap``, ``ci_upper_bootstrap``

        Sorted by ``probability`` in descending order.
    """
    logger.info(
        "Computing combined CIs (n_sim=%d, conf=%.2f, n_boot=%d)",
        n_simulations,
        confidence,
        n_bootstrap,
    )

    # Count championships per team
    counts = pd.Series(winners).value_counts()
    unique_teams: List[str] = counts.index.tolist()

    # --- Wilson CI ---
    wilson_rows: List[dict] = []
    for team in unique_teams:
        wins: int = int(counts[team])
        p_hat, ci_lo_w, ci_hi_w = compute_wilson_ci(
            wins, n_simulations, confidence
        )
        wilson_rows.append(
            {
                "team": team,
                "championships": wins,
                "probability": p_hat,
                "ci_lower_wilson": ci_lo_w,
                "ci_upper_wilson": ci_hi_w,
            }
        )

    wilson_df = pd.DataFrame(wilson_rows)

    # --- Bootstrap CI ---
    bootstrap_df = compute_bootstrap_ci(
        winners,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        random_seed=random_seed,
    )

    # --- Merge ---
    combined = wilson_df.merge(
        bootstrap_df[["team", "ci_lower_bootstrap", "ci_upper_bootstrap"]],
        on="team",
        how="left",
    )

    combined.sort_values("probability", ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        "Combined CI table ready – %d teams, top: %s (%.4f)",
        len(combined),
        combined.iloc[0]["team"] if len(combined) > 0 else "N/A",
        combined.iloc[0]["probability"] if len(combined) > 0 else 0.0,
    )

    return combined
