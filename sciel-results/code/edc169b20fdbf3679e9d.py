import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute LCB (to be maximized)
    beta = 2.0
    lcb_max = beta * sigma - mu
    
    # Normalize to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    lcb_min = np.min(lcb_max)
    lcb_max_val = np.max(lcb_max)
    lcb_range = lcb_max_val - lcb_min
    if lcb_range < 1e-9:
        lcb_norm = np.ones_like(lcb_max)
    else:
        lcb_norm = (lcb_max - lcb_min) / lcb_range
        
    # Hybrid score
    score = 0.5 * ei_norm + 0.5 * lcb_norm
    
    return score
