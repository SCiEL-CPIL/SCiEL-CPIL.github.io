import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Temperature-scaled EI: explore more when f_best is large
    # Scale factor: when f_best is large, use more exploration
    # When f_best is close to 0, use standard EI (exploitation)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add exploration bonus scaled by how far f_best is from 0
    # f_best starts around 10-20 for Beale, decreases to ~0
    # Use a smooth transition
    exploration_weight = np.clip(f_best / 10.0, 0.1, 1.0)
    exploration_bonus = exploration_weight * sigma_safe * 0.5
    
    return ei + exploration_bonus
