"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to enforce deadline
_start_time = time.time()
# Safety margin: return EI if we get close to limit
# The evaluator kills at 330s. We should return well before that.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    Replaces the exploratory blend with a purely exploitative metric
    to aggressively target the current best region.
    """
    # Check deadline
    if time.time() - _start_time > 300:
        # Fallback: return PI (or EI, but PI is the current strategy)
        # Using PI as fallback for consistency with the strategy
        z = (f_best - mu) / (sigma + 1e-9)
        z_clipped = np.clip(z, -10, 10)
        pi = norm.cdf(z_clipped)
        return np.where(np.isfinite(pi), pi, 0.0)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Probability of Improvement
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    z = (f_best - mu) / sigma
    
    # Compute PI
    # For numerical stability, clip z to avoid issues in cdf (though cdf is stable,
    # clipping prevents extreme values if sigma is tiny and mu is far)
    z_clipped = np.clip(z, -10, 10)
    
    pi = norm.cdf(z_clipped)
    
    # Ensure finite output
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
# EVOLVE-BLOCK-END
