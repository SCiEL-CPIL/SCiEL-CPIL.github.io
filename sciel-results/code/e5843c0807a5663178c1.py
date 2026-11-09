"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid EI+UCB acquisition function.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # UCB (Upper Confidence Bound) for minimization
    # We want to explore where uncertainty is high
    # UCB = mu - kappa * sigma (lower is better for minimization)
    # But we want higher acquisition scores for better candidates
    # So: acquisition_ucb = -mu + kappa * sigma
    kappa = 1.0  # Exploration-exploitation trade-off
    ucb_score = -mu + kappa * sigma_safe
    
    # Combine EI and UCB
    # Normalize both to comparable scales
    # EI is in objective units, UCB is also in objective units
    # Weight them equally
    combined = ei + ucb_score
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    score = combined / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
