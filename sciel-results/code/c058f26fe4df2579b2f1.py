"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) for minimization.
    Falls back to Lower Confidence Bound (LCB) if EI is flat.
    """
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle any non-finite values
    # If mu is not finite, assume it's as bad as f_best
    mu = np.where(np.isfinite(mu), mu, f_best)
    # If sigma is not finite or <= 0, assume very small uncertainty
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI is theoretically non-negative, but numerical errors might cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Check if the GP is confident that no improvement is possible (EI is flat)
    # In this case, switch to exploration using LCB
    # LCB score to maximize: -mu + beta * sigma
    # We use beta = 2.0 as a reasonable exploration constant for 1D
    if np.max(ei) < 1e-6:
        beta = 2.0
        lcb_scores = -mu + beta * sigma
        # Ensure finite
        lcb_scores = np.where(np.isfinite(lcb_scores), lcb_scores, 0.0)
        return lcb_scores

    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
