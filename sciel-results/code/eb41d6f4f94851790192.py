"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) with quadratic cost penalty.
    # PI is more robust to sigma scaling issues than EI in some landscapes.
    # The quadratic cost penalty aggressively favors low-cost points,
    # allowing more evaluations within the fixed budget.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    pi = norm.cdf(z)
    
    # Cost-aware score
    score = pi / (cost ** 2)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
