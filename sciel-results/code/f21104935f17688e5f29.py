import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    # sigma is a standard deviation, so it should be >= 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best < mu, z is negative.
    # If f_best > mu, z is positive.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores as required by the contract
    # If any score is nan or inf, replace it with a very small negative number
    # so that it is not selected (since higher score is better for selection)
    # Actually, EI is always >= 0. If mu is very low, EI is 0.
    # If there are numerical issues, setting to -1e9 ensures they are ignored.
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
