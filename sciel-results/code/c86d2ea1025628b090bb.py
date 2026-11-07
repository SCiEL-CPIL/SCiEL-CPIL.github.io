"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we return within wall-clock limits.
    # For a single call, this is trivial, but we keep it for safety.
    t0 = time.time()
    
    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Handle potential numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure we don't exceed a reasonable time (though this is very fast)
    # If this were a heavy computation, we'd check time here.
    # For EI on 2000 points, it's sub-millisecond.
    
    return ei
# EVOLVE-BLOCK-END
