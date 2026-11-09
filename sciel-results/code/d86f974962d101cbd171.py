import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # This is a distinct approach from the current UCB (linear combination).
    
    # Ensure sigma is positive for division
    # Using a small epsilon to prevent division by zero
    eps = 1e-10
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI formula for minimization
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # EI should be non-negative
    # Due to numerical precision, small negative values might occur, so clip to 0
    ei = np.maximum(ei, 0)
    
    return ei
