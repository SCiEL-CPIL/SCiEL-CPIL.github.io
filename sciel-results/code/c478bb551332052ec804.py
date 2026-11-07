import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use sqrt(cost) to reduce the penalty of expensive regions
    # This aligns with the "lower cost penalty" insight
    effective_cost = np.sqrt(cost)
    
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle sigma = 0 to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF
    # Using scipy.stats.norm
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    score = ei / effective_cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
