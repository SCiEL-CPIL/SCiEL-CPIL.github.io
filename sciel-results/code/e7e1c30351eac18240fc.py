"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _START_TIME > _TIMEOUT:
        # Fallback: Return EI scores (fastest reliable metric)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    # Calculate Expected Improvement (EI) for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Hybrid Score: EI + Exploration Bonus
    # The exploration bonus encourages visiting points with high uncertainty (high sigma),
    # which helps escape local optima in the early stages of optimization.
    # As sigma decreases in well-explored regions, the EI term dominates,
    # shifting the strategy towards exploitation.
    exploration_weight = 0.5
    score = ei + exploration_weight * sigma_safe
    
    return score
# EVOLVE-BLOCK-END
