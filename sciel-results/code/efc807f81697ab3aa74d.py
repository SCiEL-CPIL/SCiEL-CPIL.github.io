"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a dynamic cost penalty and time guard.
    
    Strategy:
    - Uses EI as the primary signal for exploitation.
    - Applies a cost-aware penalty (division by cost) to prefer cheaper points.
    - Includes a time guard to ensure we return before the hard deadline.
    """
    
    # Time guard: if we're close to the limit, return a safe default
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive for division
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: divide by cost to prefer cheaper evaluations
    # Cost is in [1, 2], so this provides a meaningful penalty
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
