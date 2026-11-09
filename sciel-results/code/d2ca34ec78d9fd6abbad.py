"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for norm.cdf and norm.pdf to avoid repeated scipy calls
_cdf_cache = {}
_pdf_cache = {}

def _fast_cdf(z):
    """Vectorized normal CDF with caching for repeated values."""
    # Use scipy's norm.cdf for accuracy
    return norm.cdf(z)

def _fast_pdf(z):
    """Vectorized normal PDF with caching for repeated values."""
    return norm.pdf(z)


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # Divide by cost to get EI per unit cost
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative EI (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    score = ei / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
