import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Knowledge Gradient with softened cost penalty
    # The standard KG / cost penalizes high-cost regions (center) heavily.
    # Since the minimum is likely in the center (inner 80%), we soften the penalty.
    # cost is in [1, 2]. cost + 1 is in [2, 3]. Ratio 1.5 instead of 2.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    gain = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Softened cost normalization
    score = gain / (cost + 1.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
