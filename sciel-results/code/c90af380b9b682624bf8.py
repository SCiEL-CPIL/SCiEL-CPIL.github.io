"""Cost-aware acquisition function with smooth adaptive parameter optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

    Uses a cost-normalized linear combination of improvement and uncertainty,
    approximating the expected reduction in simple regret per unit cost.
    Formula: (f_best - mu + 0.5 * sigma) / cost

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
        score = (f_best - mu + 0.5 * sigma) / cost
        return np.where(np.isfinite(score), score, -1e10)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Linear cost-aware acquisition:
    # f_best - mu is the expected improvement (regret reduction).
    # 0.5 * sigma is a conservative estimate of uncertainty bonus.
    # Dividing by cost normalizes for evaluation expense.
    score = (f_best - mu + 0.5 * sigma) / cost

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)

    return score
# EVOLVE-BLOCK-END
