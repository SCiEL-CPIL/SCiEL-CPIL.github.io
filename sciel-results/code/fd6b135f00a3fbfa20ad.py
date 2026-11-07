"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with a small exploration parameter xi
    # This encourages exploring regions with high uncertainty that might contain
    # values significantly better than f_best.
    xi = 0.01
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu - xi) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return ei
# EVOLVE-BLOCK-END
