import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Hybrid Acquisition: 80% EI, 20% Exploration (Sigma)
    # This balances exploitation (EI) and exploration (Sigma)
    # Both are in objective units.
    # We divide by cost to make it cost-aware.
    
    # Weights
    w_exploit = 0.8
    w_explore = 0.2
    
    raw_score = w_exploit * ei + w_explore * sigma
    
    # Cost-aware score
    # Cost is given as input, shape (2000,)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = raw_score / cost_safe
    
    # Return finite scores
    # Replace any non-finite values with a very small number or 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
