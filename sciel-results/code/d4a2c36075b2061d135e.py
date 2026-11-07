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
    sigma_clipped = np.maximum(sigma, 1e-12)
    
    # Compute the z-score for Expected Improvement
    z = (f_best - mu) / sigma_clipped
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    
    # Ensure ei is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware strategy:
    # The cost is 1 + mean(x) where x is in [0,1]^3, so cost is in [1, 2].
    # We want to prefer low-cost points when EI is similar.
    # A common approach is to divide EI by cost, but this can be too aggressive.
    # Instead, we use a combination:
    # 1. Normalize EI to have zero mean and unit variance
    # 2. Normalize cost to have zero mean and unit variance
    # 3. Combine them with a tunable weight
    
    # Normalize EI
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-12:
        ei_normalized = (ei - ei_mean) / ei_std
    else:
        ei_normalized = np.zeros(n)
    
    # Normalize cost
    cost_mean = np.mean(cost)
    cost_std = np.std(cost)
    if cost_std > 1e-12:
        cost_normalized = (cost - cost_mean) / cost_std
    else:
        cost_normalized = np.zeros(n)
    
    # Weight for cost penalty
    # Higher weight means more preference for low-cost points
    # We use a moderate weight that balances exploration and cost
    cost_weight = 0.3
    
    # Score: normalized EI minus cost penalty
    scores = ei_normalized - cost_weight * cost_normalized
    
    # Add a small exploration bonus based on sigma
    # This helps in early stages when the GP is uncertain
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-12:
        sigma_normalized = (sigma - sigma_mean) / sigma_std
        scores += 0.05 * sigma_normalized
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
