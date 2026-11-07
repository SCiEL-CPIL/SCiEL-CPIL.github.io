"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # Clip z to avoid numerical underflow/overflow in norm.cdf/pdf
    # For z > 8, norm.cdf is ~1.0 and norm.pdf is ~0.
    # For z < -8, norm.cdf is ~0.0 and norm.pdf is ~0.
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    
    # Standard Expected Improvement formula
    # We want to minimize f, so EI measures how much better mu could be than f_best
    # Score = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
