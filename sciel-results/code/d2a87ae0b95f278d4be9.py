# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (clip small negative values from numerical errors)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get EI per unit cost
    # cost has shape (2000,), each element is the cost of evaluating that candidate
    # cost = 1 + mean(x scaled to [0, 1]), so cost is in [1, 2]
    scores = ei / (cost + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
