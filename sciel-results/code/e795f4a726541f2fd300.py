"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Variance-Gated Expected Improvement (VEI) acquisition function.
    
    Scales standard EI by (sigma / sigma_max)^gamma to penalize
    exploitation in low-uncertainty regions and encourage exploration
    in areas where the GP is still uncertain.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Standard EI computation
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Variance gating: scale EI by (sigma / sigma_max)^gamma
    sigma_max = np.max(sigma)
    if sigma_max <= eps:
        # All sigmas are essentially zero; fall back to pure exploitation
        return ei
    
    gamma = 1.5
    sigma_ratio = sigma / (sigma_max + eps)
    gate = np.power(sigma_ratio, gamma)
    
    # Combine: weighted mix of EI and variance-gated EI
    # This ensures we still have some exploitation signal
    scores = ei * (0.5 + 0.5 * gate)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
