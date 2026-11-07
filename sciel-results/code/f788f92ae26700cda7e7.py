import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Add small epsilon to sigma to avoid division by zero
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    # norm.cdf is stable, norm.pdf underflows for large |z|
    # Clipping to [-10, 10] is safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost-aware score
    # Higher score is better
    # Avoid division by zero (cost is min 1, but just in case)
    cost_safe = np.maximum(cost, eps)
    
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
