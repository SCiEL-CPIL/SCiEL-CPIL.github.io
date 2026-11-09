"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization.
    # We want to minimize f(x). The Lower Confidence Bound is mu - beta * sigma.
    # We prioritize candidates with low LCB and low cost.
    # Score = (beta * sigma - mu) / cost.
    # Higher score means more promising (lower expected value relative to cost).
    
    beta = 2.0
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate UCB score for minimization
    # We maximize the potential for improvement: high sigma (exploration) and low mu (exploitation)
    score = (beta * sigma - mu) / cost
    
    return score
# EVOLVE-BLOCK-END
