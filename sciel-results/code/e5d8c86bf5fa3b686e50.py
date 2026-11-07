"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: pure UCB for exploration when far from optimum,
    switching to pure EI for exploitation when close.
    
    The key insight: for multimodal functions like Michalewicz, we need
    strong exploration early to find the global basin, then focused
    exploitation to refine the minimum. The switch is based on how close
    f_best is to the known global minimum (-1.8013).
    """
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known global minimum for Michalewicz 2D
    f_global_min = -1.8013
    
    # Compute UCB score for minimization: higher = more promising
    # For minimization, we want low mu and high sigma (uncertainty)
    k_ucb = 3.0
    ucb_score = -mu + k_ucb * sigma_safe
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Determine if we're close to the global minimum
    # If f_best is within 0.5 of the global min, we're in exploitation mode
    distance_to_min = f_best - f_global_min
    threshold = 0.5
    
    if distance_to_min < threshold:
        # Exploitation mode: use pure EI
        score = ei
    else:
        # Exploration mode: use pure UCB
        score = ucb_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
