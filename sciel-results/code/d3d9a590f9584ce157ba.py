"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI)
    # For minimization, we want P(f(x) < f_best)
    # This is Phi((f_best - mu) / sigma)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values where norm.cdf becomes inaccurate
    z_clipped = np.clip(z, -50, 50)
    
    pi = norm.cdf(z_clipped)
    
    # Ensure finite output
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
# EVOLVE-BLOCK-END
