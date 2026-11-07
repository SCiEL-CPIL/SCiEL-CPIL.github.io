"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track time to ensure we stay within the 330s limit
_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _start_time > _TIME_LIMIT:
        # Return zeros or simple heuristic if out of time
        return np.zeros_like(mu)

    # Cost-aware Lower Confidence Bound (LCB)
    # LCB = -(mu - kappa * sigma)
    # We want to maximize this, so we compute -mu + kappa * sigma
    # Then normalize by cost: score = (-mu + kappa * sigma) / cost
    
    # kappa = 1.0 as per directive
    kappa = 1.0
    
    # Compute LCB scores
    scores = (-mu + kappa * sigma) / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
