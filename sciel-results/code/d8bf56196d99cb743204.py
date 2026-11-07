import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute the improvement term
    improvement = f_best - mu
    
    # Standardize
    z = improvement / sigma
    
    # Use scipy for accurate CDF and PDF
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = improvement * cdf + sigma * pdf
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Higher EI means better candidate
    scores = ei
    
    return scores
