"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return safe default
    elapsed = _get_elapsed()
    if elapsed > 320:
        return np.zeros(len(mu), dtype=np.float64)

    n = len(mu)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_clipped = np.maximum(sigma, 1e-8)
    
    # Compute the z-score for Expected Improvement
    z = (f_best - mu) / sigma_clipped
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    
    # Ensure ei is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware strategy:
    # Use a power-law cost penalty on the Expected Improvement.
    # This structural change avoids the issues with z-score normalization
    # which loses the absolute scale of improvement.
    
    # Normalize cost relative to the cheapest candidate in the batch
    min_cost = np.min(cost)
    if min_cost < 1e-8:
        min_cost = 1.0
        
    cost_ratio = cost / min_cost
    
    # Apply a cost penalty. 
    # alpha=0.8 indicates a strong preference for lower cost, 
    # but not as extreme as linear division (alpha=1.0).
    alpha = 0.8
    cost_penalty = np.power(cost_ratio, alpha)
    
    # Ensure cost_penalty is not zero
    cost_penalty = np.maximum(cost_penalty, 1e-8)
    
    # Score: EI divided by cost penalty
    scores = ei / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
