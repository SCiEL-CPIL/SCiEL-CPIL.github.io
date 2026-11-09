import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        sigma_safe = np.clip(sigma, 1e-9, None)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma for numerical stability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment:
    # Cost ranges from 1 (when x is at lower bound) to 2 (when x is at upper bound)
    # We want to prefer lower-cost evaluations
    # Use a multiplicative penalty that's stronger for higher costs
    
    # Normalize cost to [0, 1] range where 0 = cheapest (cost=1), 1 = most expensive (cost=2)
    cost_norm = (cost - 1.0)  # Now in [0, 1]
    
    # Apply exponential penalty: exp(-lambda * cost_norm)
    # Higher lambda = more aggressive cost avoidance
    lambda_cost = 2.5
    
    penalty = np.exp(-lambda_cost * cost_norm)
    
    scores = ei * penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
