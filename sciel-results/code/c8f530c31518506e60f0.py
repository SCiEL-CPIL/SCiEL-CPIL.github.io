import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Deterministic Lower Confidence Bound (LCB) acquisition with high exploration.
    For minimization: score = -mu + kappa * sigma
    Higher score means more promising (lower predicted value or higher uncertainty).
    """
    # Ensure sigma is at least a small positive value to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB for minimization: we want to minimize f, so we maximize (-mu + kappa*sigma)
    # High kappa encourages exploration of uncertain regions
    kappa = 3.0
    
    score = -mu + kappa * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
