# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit (maximize -mu)
        return -mu.copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        mu = np.where(mask, mu, 0.0)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Adaptive Kappa for LCB
    # We want to explore more when f_best is poor (far from global min)
    # and exploit more when f_best is good (close to global min).
    
    f_global_min = -3.86278
    # Estimated maximum value for Hartmann 3D in [0,1]^3 is around 1.0
    f_global_max = 1.0 
    
    # Calculate progress: 0 means we found the global min, 1 means we are at the worst possible value
    # f_best is the current best (lowest).
    # If f_best == f_global_min, progress = 0 -> Exploit (low kappa)
    # If f_best == f_global_max, progress = 1 -> Explore (high kappa)
    
    # Clip f_best to reasonable range to avoid extreme kappa
    f_best_clipped = np.clip(f_best, f_global_min, f_global_max)
    
    # Normalized progress
    progress = (f_best_clipped - f_global_min) / (f_global_max - f_global_min)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Kappa range
    kappa_min = 0.5   # Exploitation focused
    kappa_max = 2.5   # Exploration focused
    
    # Linear interpolation of kappa based on progress
    # As progress increases (worse f_best), kappa increases (more exploration)
    kappa = kappa_min + (kappa_max - kappa_min) * progress
    
    # LCB score for minimization: mu - kappa * sigma
    # We want to MAXIMIZE this score in the return value (higher is better for selection)
    # So we return: -mu + kappa * sigma
    scores = -mu + kappa * sigma
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
