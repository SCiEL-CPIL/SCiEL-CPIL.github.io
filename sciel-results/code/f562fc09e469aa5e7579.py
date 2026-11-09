"""Cost-aware acquisition function using Expected Improvement with probability-of-improvement weighting."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 325.0  # Safety margin of 5s before the 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function that combines Expected Improvement (EI)
    with Probability of Improvement (PI) weighting, normalized by cost.
    
    The score is: (w * PI * EI + (1-w) * EI) / cost
    
    This structural change from pure EI/cost to a PI-weighted composite
    better balances exploitation (likely improvements) with cost efficiency.
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    # Deadline guard: If we are close to the limit, return a fast, conservative score
    # to ensure we finish safely. -mu/cost prioritizes exploitation of the current
    # best region while respecting cost.
    if time.time() > _DEADLINE - 2.0:
        cost_safe = np.where(cost > 1e-9, cost, 1.0)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    
    # Avoid division by zero in sigma for z calculation
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)

    # Compute Expected Improvement (EI) for minimization
    # Improvement is defined as f_best - f(x)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of improvement: P(f(x) < f_best)
    pi = norm.cdf(z_clipped)
    
    # Expected improvement
    ei = (f_best - mu) * pi + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Composite score: weighted combination of PI*EI and EI, divided by cost
    # PI*EI rewards candidates that are BOTH likely to improve AND have high expected improvement
    # Pure EI/cost rewards cost-efficiency
    # Weight of 0.6 on the PI-weighted term balances exploitation with cost efficiency
    w = 0.6
    scores = (w * pi * ei + (1.0 - w) * ei) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores for stability
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
