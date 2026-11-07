"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    Probability of Improvement (PI) with an adaptive exploration weight.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: blend EI with PI*sigma
    # Early on (high uncertainty), explore more; later, exploit more.
    # Use the mean sigma as a proxy for exploration phase.
    mean_sigma = np.mean(sigma_safe)
    # If mean_sigma is large, we're in early exploration phase
    # Scale exploration weight inversely with progress
    # When f_best is close to 0 (minimum), reduce exploration
    # Beale minimum is 0, so f_best approaching 0 means we're close
    
    # Adaptive beta: higher when f_best is large (early), lower when f_best is small (late)
    # f_best ranges from potentially large values down to 0
    # Use a smooth transition
    if f_best > 1.0:
        beta = 0.3  # More exploration early
    elif f_best > 0.1:
        beta = 0.15  # Moderate exploration
    else:
        beta = 0.05  # Less exploration when close to optimum
    
    # Exploration term: PI-weighted sigma
    exploration = beta * pi * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
