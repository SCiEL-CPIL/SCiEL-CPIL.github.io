"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for minimization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    if _check_deadline():
        # Fallback: return deterministic EI if we're running out of time
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # The Lower Confidence Bound (LCB) of f(x) is mu - beta * sigma.
    # We want to find the minimum, so we want the lowest LCB.
    # Maximizing the acquisition score is equivalent to minimizing the LCB.
    # Score = -LCB / cost = -(mu - beta * sigma) / cost = (-mu + beta * sigma) / cost
    
    # Beta = 1.0 balances exploration and exploitation.
    # Higher beta encourages exploration (higher sigma).
    # Lower beta encourages exploitation (lower mu).
    beta = 1.0
    
    sigma_safe = sigma + 1e-9
    cost_safe = cost + 1e-9
    
    scores = (-mu + beta * sigma_safe) / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
