"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3d.
    
    Uses a robust Expected Improvement (EI) variant with numerical safeguards.
    For minimization, EI is computed as:
        EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    We add small safeguards for numerical stability and ensure finite outputs.
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues
    z_clipped = np.clip(z, -50.0, 50.0)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure non-negative EI (EI should be >= 0 by definition, but numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
