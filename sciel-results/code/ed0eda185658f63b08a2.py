"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with safety margin
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit
    
    # Compute Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Check if we're close to deadline - if so, return immediately
    if time.time() > deadline - 5:
        # Ensure scores are finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    return scores
# EVOLVE-BLOCK-END
