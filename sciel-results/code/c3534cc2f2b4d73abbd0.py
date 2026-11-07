import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition: Expected Improvement + Linear Exploration (LCB-like)
    """
    start_time = time.time()
    deadline = 325.0

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # EI for minimization
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Exploration term: Encourage high uncertainty
    # Using a coefficient to balance with EI
    exploration_weight = 1.5
    exploration = exploration_weight * sigma_safe
    
    scores = ei + exploration
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    if time.time() - start_time > deadline:
        return scores
        
    return scores
