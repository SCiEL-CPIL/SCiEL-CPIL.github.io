# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best observed value so far (minimum)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better)
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Handle edge cases: ensure sigma is positive and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Replace any non-finite values in mu and sigma with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Compute Local Intensive Expected Improvement (LIEI) for minimization
    # First, compute the raw improvement term
    improvement = f_best - mu
    
    # Clip improvement to be non-negative (LIEI core: suppress exploration in
    # regions where GP predicts values worse than current best)
    improvement_clipped = np.maximum(improvement, 0.0)
    
    # z = (f_best - mu) / sigma
    z = improvement / sigma
    
    # Clip z to avoid extreme values that can cause numerical issues
    z = np.clip(z, -100, 100)
    
    # LIEI: use clipped improvement in the formula
    # LIEI = (f_best - mu)_+ * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    liei = improvement_clipped * cdf_z + sigma * pdf_z
    
    # Ensure all values are finite
    liei = np.where(np.isfinite(liei), liei, 0.0)
    
    # For regions where improvement is negative (GP predicts worse than best),
    # the LIEI reduces to sigma * phi(z), which is small but positive.
    # This naturally deprioritizes those regions while maintaining smoothness.
    ei = liei
    
    # Check time before returning
    if time.time() > deadline:
        return ei
    
    return ei
# EVOLVE-BLOCK-END
