"""Cost-aware UCB acquisition for Styblinski-Tang 2D BO, with a wall-clock guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 300.0          # hard wall-clock limit
_SAFETY = 25.0             # return early with margin
_T0 = time.perf_counter()


def _deadline_hit():
    return (time.perf_counter() - _T0) > (_DEADLINE - _SAFETY)


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware UCB acquisition.

    mu, sigma, cost: shape (2000,). sigma is a standard deviation.
    Returns 2000 finite scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()

    if _deadline_hit():
        # Exploit the current best prediction, cost-adjusted.
        z = (f_best - mu) / (np.maximum(sigma, 1e-9) + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / np.maximum(cost, 1e-9)

    n = mu.size

    # Standardize sigma to a 0..1 scale (robust to GP noise floor).
    sig = np.maximum(sigma, 1e-9)
    s_med = float(np.median(sig))
    s_scale = max(s_med, 1e-9)

    # Cost-aware exploration budget: spend more exploration early, taper off.
    # cost is cumulative (>= ~5 at start, grows toward 30).
    frac = 1.0 - min(max((float(np.median(cost)) - 5.0) / 25.0, 0.0), 1.0)
    beta = 0.6 + 1.6 * frac          # 0.6 (late) .. 2.2 (early)

    # UCB: lower mu is better (minimization), so score = -mu + beta * sigma.
    # Normalize sigma by its median so the term is in objective units.
    ucb = -mu + beta * (sig / s_scale)

    # Penalize expensive evaluations so we favor cheap points.
    # cost in [1,2]; use a mild multiplicative discount.
    cost_disc = np.maximum(cost, 1e-9)
    score = ucb / cost_disc

    # Guard against non-finite values.
    if not np.all(np.isfinite(score)):
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
# EVOLVE-BLOCK-END
