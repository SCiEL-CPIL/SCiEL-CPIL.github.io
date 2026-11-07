"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT_LIMIT = 290.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best, cost):
    # Time guard: if we are close to the deadline, return a simple score
    # to ensure quick termination of the current step or to prioritize exploitation.
    # We prioritize exploitation (lowest mu) to avoid wasting time on exploration.
    if time.time() - _start_time > _TIMEOUT_LIMIT:
        # Return scores favoring low mu (best prediction) and low cost
        # Since we want to maximize the score, we use -mu.
        # Cost is in denominator.
        return -mu / cost

    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # norm.cdf(-10) is ~0, norm.cdf(10) is ~1
    # norm.pdf(10) is ~0
    # It's safe, but let's be explicit.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score
    # Ensure cost is positive (it is, min 1)
    score = ei / cost
    
    # Handle any potential NaNs or Infs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
