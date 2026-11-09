"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    This approach calculates the expected improvement over the current best
    observed value. It is structurally different from LCB methods as it
    integrates the posterior distribution to estimate the expected gain.
    
    Score = EI(mu, sigma, f_best)
    Higher scores indicate candidates with higher expected improvement.
    
    Args:
        mu: Array of GP mean predictions (2000,).
        sigma: Array of GP standard deviation predictions (2000,).
        f_best: The lowest observed function value.
        
    Returns:
        Array of EI scores (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    if time.time() > deadline:
        return np.zeros(n)
    
    # Sanitize inputs
    eps = 1e-8
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_clean = np.maximum(sigma_clean, eps)
    
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    improvement = f_best - mu_clean
    z = improvement / sigma_clean
    
    # Handle numerical stability for large z
    # For very large positive z, Phi(z) ~ 1, phi(z) ~ 0
    # For very large negative z, Phi(z) ~ 0, phi(z) ~ 0
    # We can just compute directly, scipy handles it well.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = improvement * cdf_z + sigma_clean * pdf_z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
