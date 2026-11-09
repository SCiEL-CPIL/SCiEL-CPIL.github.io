"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Deadline guard: return fallback if we're getting close to timeout
    deadline = start_time + 290  # 290s margin from 300s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scoring:
    # Use EI divided by cost to balance improvement potential against evaluation cost
    # This encourages finding improvements in cheap regions
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Fallback to EI if computation takes too long (though it shouldn't)
        return ei
    
    return score
# EVOLVE-BLOCK-END
