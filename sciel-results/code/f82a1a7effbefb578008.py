"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for time management
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we are too close to the limit
    elapsed = time.time() - _start_time
    if elapsed > 295:
        # Return a simple deterministic score to finish quickly
        # Prefer points with low cost and low mean (optimistic)
        return (-mu + 0.5 * sigma) / np.maximum(cost, 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-Aware Score: Divide by cost
    # Higher score = better candidate
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
