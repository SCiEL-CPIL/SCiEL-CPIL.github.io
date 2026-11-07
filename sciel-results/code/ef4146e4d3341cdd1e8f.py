"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Use a dynamic threshold for EI to balance exploration/exploitation
    # The threshold is scaled by the current best value
    # For Weierstrass, f_best starts near 0 and decreases
    # We use a small epsilon to avoid division by zero and to encourage exploration initially
    eps = 1e-6
    threshold = f_best - eps * np.maximum(np.abs(f_best), 1.0)
    
    # Expected Improvement for minimization with threshold
    # EI = (threshold - mu) * Phi(z) + sigma * phi(z)
    # where z = (threshold - mu) / sigma
    z = (threshold - mu) / sigma
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI
    ei = (threshold - mu) * cdf_z + sigma * pdf_z
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
