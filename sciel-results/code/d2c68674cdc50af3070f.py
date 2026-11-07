"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive behavior
_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    _call_count += 1
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to standard EI/Cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Calculate Expected Improvement (EI)
    # mu, sigma are arrays of shape (2000,)
    # f_best is scalar
    # cost is array of shape (2000,)
    
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Uncertainty Score (U)
    # For minimization, we want low mu and high sigma.
    # Score = -mu + beta * sigma
    beta = 1.0
    u_score = -mu + beta * sigma
    
    # Adaptive weight
    # Assume max ~20 calls
    max_calls = 20
    # w is weight for EI (exploitation)
    # Start with 0.2 (80% exploration), end with 0.9 (90% exploitation)
    progress = min((_call_count - 1) / (max_calls - 1), 1.0)
    w = 0.2 + 0.7 * progress
    
    # Combine
    # We want to maximize the acquisition score
    acq = w * ei + (1 - w) * u_score
    
    # Cost adjustment
    # Divide by cost
    scores = acq / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
