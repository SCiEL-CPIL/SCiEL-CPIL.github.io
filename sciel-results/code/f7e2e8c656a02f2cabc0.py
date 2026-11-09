import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    Z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, -1e10)
    
    return scores
