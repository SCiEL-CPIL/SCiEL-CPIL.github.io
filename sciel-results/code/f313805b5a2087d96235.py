"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # seconds, with safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety: return simple expected improvement to avoid timeout
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Knowledge gradient approximation with exploration bonus
    # Structural change: use a hybrid of probability of improvement 
    # and expected improvement with adaptive exploration
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    p_imp = norm.cdf(z)
    
    # Expected improvement
    ei = (f_best - mu) * p_imp + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward high uncertainty regions
    # Use a tempered exploration term that decays as we gain confidence
    exploration_weight = 0.3
    exploration = exploration_weight * sigma_safe
    
    # Combine: weighted mix of EI and exploration
    # This creates a different optimization landscape than pure EI
    score = 0.7 * ei + 0.3 * exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
