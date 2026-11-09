"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = time.time() + 325.0  # safety margin before 330s kill
_MIN_COST = 1.0
_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining UCB exploration with expected improvement,
    normalized by evaluation cost.

    Parameters
    ----------
    mu : ndarray, shape (N,)
        GP mean predictions at N candidate points.
    sigma : ndarray, shape (N,)
        GP standard deviation predictions at N candidate points.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (N,)
        Cost of evaluating each candidate.

    Returns
    -------
    scores : ndarray, shape (N,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp cost to minimum 1.0
    cost = np.maximum(cost, _MIN_COST)

    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, _EPS)

    # UCB term: exploit low mu, explore high sigma
    # Beta controls exploration; 2.0 gives a good balance for Griewank
    beta = 2.0
    ucb = mu - beta * sigma_safe

    # Expected improvement term (for minimization)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Combine: weighted sum of UCB and EI, both divided by cost
    # UCB already has the right sign (lower mu, higher sigma => higher ucb score)
    # We want to maximize, so we negate mu contribution appropriately
    # Actually UCB for minimization: score = -mu + beta*sigma (higher is better)
    # Let me redo: for minimization, we want low mu and high sigma
    # UCB score (higher = better) = -mu + beta * sigma
    # EI score (higher = better) = EI value

    # Recompute UCB correctly for minimization
    ucb_score = -mu + beta * sigma_safe

    # Normalize both by cost
    ucb_norm = ucb_score / cost
    ei_norm = ei / cost

    # Weighted combination: 0.5 * UCB + 0.5 * EI
    # This balances exploration (UCB) and exploitation (EI)
    scores = 0.5 * ucb_norm + 0.5 * ei_norm

    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
