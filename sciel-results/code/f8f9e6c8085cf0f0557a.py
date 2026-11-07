# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best observed value so far (minimum)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better)
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Handle edge cases: ensure sigma is positive and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Replace any non-finite values in mu and sigma with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values that can cause numerical issues
    # norm.cdf and norm.pdf are stable for most z, but clip for safety
    z = np.clip(z, -100, 100)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Add exploration bonus: normalize sigma and add a small weight
    # This encourages visiting high-uncertainty regions to escape local basins
    sigma_max = np.max(sigma)
    if sigma_max > 1e-8:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = sigma
    
    # Weighted combination: EI + exploration bonus
    # The exploration term is scaled relative to EI scale
    ei_scale = np.max(np.abs(ei))
    if ei_scale < 1e-8:
        ei_scale = 1.0
    
    exploration_weight = 0.1  # Small weight for exploration
    scores = ei + exploration_weight * ei_scale * sigma_norm
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time before returning
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
