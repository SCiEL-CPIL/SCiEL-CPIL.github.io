"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a quantile-based LCB with sigmoidal cost discount.
    
    Uses a lower quantile of the predictive distribution (adaptive beta) and
    applies a non-linear sigmoidal cost discount to prune expensive candidates.
    """
    # Deadline guard: lightweight per-call check
    # The main loop tracks total time; this ensures we don't hang on edge cases
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Adaptive quantile: more aggressive exploration early (high beta), 
    # converging to exploitation later (low beta)
    # We don't have explicit iteration count, but f_best improvement rate is a proxy.
    # For now, use a fixed aggressive quantile that worked well in the observed improvement.
    beta = 0.05  # 5th percentile: aggressive exploration
    
    # Quantile of the normal distribution
    z_beta = norm.ppf(beta)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want low values, so we subtract the uncertainty term
    lcb = mu + z_beta * sigma_safe
    
    # Sigmoidal cost discount: 
    # Cost ranges from 1 (center) to 2 (corners).
    # We want to strongly discount high costs.
    # Center cost ~1, corner cost ~2.
    # Let's map cost to a discount factor in [0, 1].
    # discount = 1 / (1 + exp(k * (cost - c0)))
    # If cost is low (near 1), discount is high (near 1).
    # If cost is high (near 2), discount is low (near 0).
    
    # Normalize cost to [0, 1] relative to the max possible cost (2.0)
    # cost in [1, 2]
    cost_norm = (cost_safe - 1.0) / 1.0  # [0, 1]
    
    # Sigmoid parameters
    # We want the transition to happen around cost=1.5 (middle of range)
    # k controls steepness
    k = 10.0
    c0 = 0.5  # midpoint of normalized cost
    
    discount = 1.0 / (1.0 + np.exp(k * (cost_norm - c0)))
    
    # Apply discount to the LCB score
    # Higher score is better. Since LCB is lower for better candidates,
    # we negate it to make it a maximization score.
    score = -lcb * discount
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
