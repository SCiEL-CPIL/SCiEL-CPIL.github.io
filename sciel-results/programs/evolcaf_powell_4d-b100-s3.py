"""Rewritten program with a stronger, direct, cost-aware acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

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
        # Return a safe, finite fallback: favor low mu / high sigma
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu / (sigma_safe + 1e-9) + np.log(sigma_safe + 1.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Clamp sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Standard z-score for expected improvement
    improvement = f_best - mu
    z = improvement / sigma_safe

    # Expected Improvement (EI)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize (f_best - mu + kappa * sigma)
    # This encourages exploration of high uncertainty regions even if mu is not optimal
    kappa = 1.5
    ucb_value = f_best - mu + kappa * sigma_safe
    ucb_value = np.where(np.isfinite(ucb_value), ucb_value, 0.0)

    # Combine EI and UCB
    # EI is good for exploitation. UCB is good for exploration.
    # We weight them to balance the two.
    # The UCB term is in the same units as EI (objective value difference).
    combined_value = ei + 0.5 * ucb_value
    
    # Cost-aware: per unit cost
    total_score = combined_value / cost

    # Ensure finiteness
    total_score = np.where(np.isfinite(total_score), total_score, -1e10)

    return total_score
# EVOLVE-BLOCK-END
