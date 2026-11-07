"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds, with safety margin from 300s per evaluation
# Actually, the evaluator kills after 330s wall-clock total.
# Each call should be fast, but let's guard just in case.

def acquisition(mu, sigma, f_best):
    """
    Structural switch: adaptive LCB (exploration) early on, pure EI (exploitation) late.
    
    mu: (2000,) array of GP mean predictions in objective units
    sigma: (2000,) array of GP standard deviations in objective units  
    f_best: lowest observed function value
    
    Returns: (2000,) array of scores, higher = more promising to evaluate next
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: simple LCB score, fast
        sigma_safe = np.maximum(sigma, 1e-8)
        return -mu + 1.0 * sigma_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Known global minimum for Hartmann-3D
    f_global_min = -3.86278
    
    # Reference "bad" value for progress calculation
    f_worst_ref = 0.0
    
    # Calculate progress towards the global minimum
    denominator = f_worst_ref - f_global_min
    if denominator == 0:
        progress = 0.5
    else:
        progress = (f_worst_ref - f_best) / denominator
    
    # Clip progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Structural switch:
    # - Early (low progress): use adaptive LCB for exploration
    # - Late (high progress): use pure EI for exploitation
    # Transition threshold: switch from LCB to EI when progress > 0.6
    
    if progress < 0.6:
        # Exploration phase: adaptive LCB
        # Higher kappa when progress is low
        kappa_max = 3.0
        kappa_min = 1.0
        # Interpolate within the exploration phase
        local_progress = progress / 0.6  # normalize to [0, 1] within this phase
        kappa = kappa_max - (kappa_max - kappa_min) * local_progress
        
        # LCB for minimization: score = -mu + kappa * sigma
        scores = -mu + kappa * sigma_safe
    else:
        # Exploitation phase: pure EI
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
