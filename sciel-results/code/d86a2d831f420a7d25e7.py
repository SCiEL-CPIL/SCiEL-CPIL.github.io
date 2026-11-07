import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses Variance-Gated Expected Improvement (VEI) for minimization.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI for minimization
    ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Variance-Gated Expected Improvement (VEI)
    # Scale EI by (sigma / sigma_max)^gamma
    # This penalizes exploitation in low-uncertainty regions
    # and encourages exploration in high-uncertainty regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        gamma = 1.5
        gate = (sigma_safe / sigma_max) ** gamma
    else:
        gate = np.ones_like(sigma_safe)
        
    scores = ei_scores * gate
    
    # Ensure all scores are finite and non-negative
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
