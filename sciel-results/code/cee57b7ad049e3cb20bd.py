"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s total wall-clock, use 325s to be safe
    # This function is called repeatedly, so we just ensure it's fast
    # The main computation is vectorized and should be < 1ms typically
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure ei is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: encourage exploration in uncertain regions
    # This helps escape local optima, especially in early iterations
    exploration_bonus = 0.1 * sigma_safe
    
    # Combined numerator
    numerator = ei + exploration_bonus
    
    # Cost-aware score: divide by cost
    # cost is shape (2000,), all values >= 1
    scores = numerator / cost
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
