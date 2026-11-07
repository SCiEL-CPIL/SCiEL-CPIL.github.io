"""Direct cost-aware acquisition for Bayesian optimization on griewank 2d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: Expected Improvement (EI) weighted by inverse cost.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions in objective units
    - sigma: array of shape (2000,) - GP std-dev predictions in objective units
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs (1 + mean of scaled coords)
    
    Returns:
    - scores: array of shape (2000,) of finite scores, higher = more promising
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement: EI(x) = E[max(f_best - f(x), 0)]
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # phi(z) is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to 0 (numerical safety, though EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Weight by inverse cost to prefer cheaper evaluations
    inv_cost = 1.0 / np.maximum(cost, 1e-9)
    
    # Combine: weighted EI
    scores = inv_cost * ei
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
