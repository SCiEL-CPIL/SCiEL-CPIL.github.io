"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Check time guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to simple heuristic if running out of time
        # Minimize mu, prefer low cost
        return -mu / cost

    # Expected Improvement
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Numerically stable EI calculation
    # For z < 0, cdf is small, pdf is large? No.
    # Standard formula is fine for typical ranges.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add UCB-like exploration term
    # beta controls exploration
    beta = 0.5
    score = (ei + beta * sigma) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
