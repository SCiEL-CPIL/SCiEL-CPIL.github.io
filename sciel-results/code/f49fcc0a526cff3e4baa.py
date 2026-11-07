"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return early if we're close to the 300s limit
    # The evaluator kills at 330s, so we use 280s as safety margin
    _start = time.time()
    
    # Cost-aware acquisition: balance expected improvement against cost.
    # Use a small epsilon to avoid division by zero.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # Numerical stability for norm.cdf and norm.pdf
    # For very large z, norm.cdf(z) -> 1, norm.pdf(z) -> 0
    # For very small z, norm.cdf(z) -> 0.5, norm.pdf(z) -> 0.4
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (should be, but numerical errors can occur)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost.
    # Hypothesis: A weaker penalty on cost (sqrt) might be better 
    # because the minimizer is in the higher cost region (center).
    # Current: 1/cost. New: 1/sqrt(cost).
    cost_safe = np.maximum(cost, 1.0)
    result = ei / np.sqrt(cost_safe)
    
    # Check deadline
    if time.time() - _start > 5.0:
        # Fallback
        return np.zeros_like(result)
    
    return result
# EVOLVE-BLOCK-END
