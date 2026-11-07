"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # UCB acquisition with adaptive beta and time guard
    start_time = time.time()
    
    # Use a slightly higher beta to encourage more exploration
    # In the early stages of BO, more exploration helps find the global minimum
    beta = 1.5
    
    # Compute UCB score: -mu + beta * sigma
    # Higher score means more likely to be evaluated next
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Time guard: if computation took too long, return what we have
    # (though for this simple computation it should be fast)
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        pass  # Already computed, just return
    
    return scores
# EVOLVE-BLOCK-END
