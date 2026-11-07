import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    eps = 1e-9
    sigma_safe = sigma + eps
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Add exploration bonus
    # Higher sigma means more exploration
    exploration_weight = 0.5
    score = ei + exploration_weight * sigma
    
    return score
