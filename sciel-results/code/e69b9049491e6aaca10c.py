"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute standard normal CDF/PDF lookup for speed (optional, but we use scipy which is fast enough)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP mean predictions.
    sigma : ndarray of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We want to maximize EI / cost (cost-aware)
    
    z = (f_best - mu) / sigma_safe
    
    # Handle numerical stability
    # For large positive z, Phi(z) ~ 1, phi(z) ~ 0 -> EI ~ f_best - mu
    # For large negative z, Phi(z) ~ 0, phi(z) ~ 0 -> EI ~ 0
    
    # Use scipy.stats.norm for accurate CDF/PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative (numerical issues might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # If cost is zero (shouldn't happen), treat as high cost
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
