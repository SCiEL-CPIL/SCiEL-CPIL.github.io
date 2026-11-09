"""Cost-aware acquisition function with POI-based approach for Powell 4D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def _smooth_transition(x, low, high):
    """Smooth sigmoid-like transition from 0 to 1 as x goes from low to high."""
    if high <= low:
        return 1.0 if x >= high else 0.0
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    # Smoothstep: 3t^2 - 2t^3
    return 3.0 * t * t - 2.0 * t * t * t


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

    Uses a cost-normalized probability of improvement (POI) combined with a
    deterministic regret lower bound. The score is (f_best - mu + beta * sigma) / cost,
    which directly optimizes expected reduction in simple regret per unit cost.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns
    -------
    np.ndarray, shape (2000,)
        Finite scores; higher = prefer to evaluate next.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 1e10)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        score = (f_best - mu + 0.5 * sigma) / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, -1e10)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)

    # Adaptive beta: exploration weight
    # When f_best is large (early), use higher beta to explore more
    # When f_best is small (late), use lower beta to exploit
    t_exploit = _smooth_transition(f_best, 0.0, 2.0)
    # beta ranges from 0.5 (exploit) to 1.5 (explore)
    beta = 0.5 + 1.0 * (1.0 - t_exploit)

    # Core POI-based score: expected improvement lower bound per unit cost
    # f_best - mu is the deterministic improvement
    # beta * sigma adds exploration via uncertainty
    score = (f_best - mu + beta * sigma_safe) / np.maximum(cost, 1e-6)

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)

    return score
# EVOLVE-BLOCK-END
