import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement (EI) for minimization
    # EI encourages both exploitation (low mu) and exploration (high sigma)
    # This serves as the diversity-promoting strategy requested.
    
    # Prevent division by zero for points with very low predictive variance
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized distance from best
    z = (f_best - mu) / sigma_safe
    
    # Compute EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF of standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure scores are non-negative
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Normalize by cost to account for evaluation budget
    # Lower cost evaluations are preferred if EI is similar
    scores = ei_scores / cost
    
    return scores
