"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values where norm.cdf and norm.pdf become inaccurate
    z_clipped = np.clip(z, -50, 50)
    
    # Compute EI using stable formulas
    # For z <= 0: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For z > 0: use the same formula but with clipped z
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
