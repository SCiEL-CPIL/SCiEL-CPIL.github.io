# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: kill the whole acquisition call if it runs too long.
# The evaluator kills after 330s wall-clock; we target finishing well before that.
_DEADLINE = 330.0
_START_TIME = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6D.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed objective value so far.
    cost : ndarray, shape (2000,)
        Evaluation cost for each candidate: 1 + mean(x) where x is in [0,1]^6.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more desirable candidates.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Replace any non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Check deadline
    if _time_left() < 2.0:
        # Emergency fallback: just return EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware: divide by cost
    # Also add a small exploration bonus to encourage diverse exploration early on
    # The cost is between 1 and 2 (since mean(x) in [0,1])
    # We weight by 1/cost to prefer cheaper evaluations

    scores = ei / cost

    # Add a small bonus for high uncertainty (sigma) to encourage exploration
    # This helps escape local optima in the 6D space
    exploration_bonus = 0.01 * sigma / cost

    scores = scores + exploration_bonus

    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
