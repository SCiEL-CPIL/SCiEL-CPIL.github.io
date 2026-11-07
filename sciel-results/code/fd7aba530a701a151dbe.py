import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero or invalid inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term: UCB-like
    # Adding a portion of sigma encourages exploration of uncertain regions.
    # This helps in finding the global minimum if initial points miss it.
    exploration_weight = 0.5
    exploration = exploration_weight * sigma_safe
    
    # Combined benefit
    benefit = ei + exploration
    
    # Cost-aware score: Higher is better
    # cost is in [1, 2]
    score = benefit / cost
    
    # Ensure finite output
    return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
