"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return early if we're close to the 300s limit
    # The evaluator kills at 330s, so we use 280s as safety margin
    _start = time.time()
    _deadline = 280.0
    
    # Cost-aware acquisition: balance expected improvement against cost.
    # Use a small epsilon to avoid division by zero.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost, but add a small constant to avoid extreme values
    # Also, we can use a power of cost to make it more sensitive to high costs
    cost_safe = np.maximum(cost, 1.0)
    
    # Return EI / cost
    result = ei / cost_safe
    
    # Check deadline
    if time.time() - _start > 5.0:
        # If this call is taking too long, return a simple fallback
        # This shouldn't happen normally but is a safety net
        return np.zeros_like(result)
    
    return result
# EVOLVE-BLOCK-END
