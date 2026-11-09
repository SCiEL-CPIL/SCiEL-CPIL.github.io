"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard
_start_time = time.time()
_TOTAL_DEADLINE = 330.0  # Seconds
_SAFETY_MARGIN = 5.0     # Seconds

def _check_deadline():
    return (time.time() - _start_time) > (_TOTAL_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    # Check if we are close to the global deadline
    if _check_deadline():
        # Return a neutral score to avoid hanging or crashing, 
        # though the evaluator should kill the process soon.
        return np.zeros_like(mu, dtype=np.float64)

    # Ensure arrays are float64 for precision
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    # Using scipy.stats.norm for stability
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Exponential penalty
    # Based on insights, exp(-cost) aligns better with Levy 2D structure
    # where the minimum is near the center (low cost).
    # Cost is in range [1, 2] typically.
    score = ei * np.exp(-cost)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
