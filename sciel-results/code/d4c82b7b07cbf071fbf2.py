"""Cost-aware acquisition for Bayesian optimization on 4 test functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: cost-aware expected improvement with exploration.
    
    Parameters:
    - mu: (2000,) array of GP means
    - sigma: (2000,) array of GP standard deviations
    - f_best: scalar, best observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of scores (higher = more desirable)
    """
    # Deadline guard: we don't have a global timer here, but the function is called
    # in a loop. We'll keep it fast.
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    score = ei / cost
    
    # Add a small exploration term based on uncertainty
    # This helps when f_best is close to the minimum
    exploration = sigma / cost
    
    # Combine: primarily EI, with some exploration
    # Weight: 0.9 * EI/cost + 0.1 * sigma/cost
    score = 0.9 * score + 0.1 * exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
