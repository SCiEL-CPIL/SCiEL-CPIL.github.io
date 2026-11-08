"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition blending EI and UCB for minimization."""
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardize for z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB-style lower confidence bound (we want to minimize, so lower is better)
    # Exploration bonus: xi * sigma
    xi = 0.3
    ucb_score = (f_best - mu) + xi * sigma_safe
    
    # Blend: 70% EI + 30% UCB
    blended = 0.7 * ei + 0.3 * ucb_score
    
    # Normalize by cost (cheaper points get higher score)
    score = blended / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
