"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _lcb_per_cost(mu, sigma, f_best, cost, kappa=1.5):
    """Lower Confidence Bound per unit cost for minimization."""
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    # LCB for minimization: we want to find low values
    # mu - kappa * sigma gives a lower bound on the true value
    # Higher LCB scores mean we are more optimistic about finding low values
    lcb = mu - kappa * sigma
    # Cost-aware: divide by cost to penalize expensive evaluations
    score = lcb / cost_safe
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using LCB.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Compute LCB per unit cost
    scores = _lcb_per_cost(mu, sigma, f_best, cost, kappa=1.5)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
