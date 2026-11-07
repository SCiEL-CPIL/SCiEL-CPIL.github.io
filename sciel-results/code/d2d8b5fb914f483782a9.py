# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: we are told the evaluator kills after 330s wall-clock.
# The acquisition function is called repeatedly. We don't know the exact time budget per call,
# but we know the total wall-clock is 330s. We should be fast.
# The current implementation is already quite fast (O(N) with N=2000).
# Let's ensure we don't do anything heavy.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: mean of GP predictions, shape (2000,)
    - sigma: std dev of GP predictions, shape (2000,)
    - f_best: best observed function value
    - cost: cost of evaluating each candidate, shape (2000,)
    
    Returns:
    - scores: acquisition values, shape (2000,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero and negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI: EI / cost
    scores = ei / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
