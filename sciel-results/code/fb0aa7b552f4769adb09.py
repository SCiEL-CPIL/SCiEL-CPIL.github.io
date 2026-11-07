# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _t0 > _DEADLINE:
        # Fallback to exploitation if out of time
        return -mu

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement (PI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure finite scores
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
# EVOLVE-BLOCK-END
