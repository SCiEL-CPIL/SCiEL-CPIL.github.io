"""Cost-aware acquisition function for Bayesian optimization (powell 4d).

Uses a hybrid of cost-normalized Expected Improvement (exploitation) and a
knowledge-gradient-style exploration bonus that scales with local uncertainty,
plus a small diversity penalty against the already-evaluated cost profile to
encourage moving away from cheap, well-sampled regions.

Includes a wall-clock deadline guard so the evaluator is never killed.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
_DEADLINE_START = time.time()
# The evaluator kills after 330 s wall-clock.  Leave a generous safety margin.
_HARD_LIMIT = 330.0 - 45.0  # seconds of our own work we allow to elapse


def _safe_z(mu, sigma, f_best):
    """z = (f_best - mu) / sigma with sigma floored to avoid div-by-zero."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return np.clip(z, -1e4, 1e4)


def acquisition(mu, sigma, f_best, cost):
    """Return 2000 finite acquisition scores (higher = evaluate next).

    Parameters
    ----------
    mu : array, shape (2000,)
        GP mean predictions in objective units.
    sigma : array, shape (2000,)
        GP standard-deviation predictions in objective units.
    f_best : float
        Lowest observed objective value so far.
    cost : array, shape (2000,)
        Cost of evaluating each candidate, in [1, 2].

    Returns
    -------
    score : array, shape (2000,)
        Finite acquisition scores; higher means "evaluate this next".
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # ------------------------------------------------------------------
    # Deadline guard: if we have burned through our allowed time, return
    # a safe, finite fallback immediately rather than risk the kill.
    # ------------------------------------------------------------------
    if (time.time() - _DEADLINE_START) > _HARD_LIMIT:
        # Fallback: prefer low mu, low cost, modest exploration.
        cost_safe = np.maximum(cost, 1e-6)
        fallback = -mu / cost_safe + 0.05 * sigma / cost_safe
        return np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)

    cost_safe = np.maximum(cost, 1e-6)

    # ------------------------------------------------------------------
    # 1) Cost-normalized Expected Improvement (exploitation)
    # ------------------------------------------------------------------
    z = _safe_z(mu, sigma, f_best)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    ei_cost = ei / cost_safe

    # ------------------------------------------------------------------
    # 2) Exploration bonus: reward high-uncertainty, cheap regions.
    #    We use a "knowledge-gradient" style term: sigma / cost, but
    #    scaled down so it does not dominate the exploitation signal.
    #    A slight super-linear boost in sigma helps escape flat regions.
    # ------------------------------------------------------------------
    sigma_safe = np.maximum(sigma, 1e-9)
    # Use a mild power to emphasise high-uncertainty regions a bit more.
    explore = (sigma_safe ** 1.1) / cost_safe
    explore *= 0.08  # small coefficient to keep it subordinate to EI

    # ------------------------------------------------------------------
    # 3) Diversity / anti-clustering penalty.
    #    Candidates that are cheap (low cost) have likely been sampled
    #    more often in the uniform-candidate pool.  A small negative
    #    term proportional to (1 / cost) discourages repeatedly picking
    #    the same cheap region, nudging the search toward more
    #    expensive (i.e. more informative, less-sampled) areas.
    # ------------------------------------------------------------------
    diversity = -0.02 / cost_safe

    # ------------------------------------------------------------------
    # 4) Combine
    # ------------------------------------------------------------------
    score = ei_cost + explore + diversity

    # Guarantee all-finite output.
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
