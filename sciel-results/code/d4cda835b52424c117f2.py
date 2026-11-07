"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Variance-Gated Expected Improvement (VEI) acquisition function.
    
    Scales standard EI by (sigma / sigma_max)^gamma to penalize exploitation
    in low-uncertainty regions and encourage exploration where the GP is uncertain.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Safety guard for very small sigma values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Variance gating: scale EI by relative uncertainty
    sigma_max = np.max(sigma_safe)
    if sigma_max < 1e-12:
        # If all sigmas are essentially zero, just return EI
        return ei
    
    # gamma = 1.5: strong preference for high-uncertainty regions
    gamma = 1.5
    variance_gate = (sigma_safe / sigma_max) ** gamma
    
    # Combine: EI weighted by uncertainty
    score = ei * variance_gate
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
