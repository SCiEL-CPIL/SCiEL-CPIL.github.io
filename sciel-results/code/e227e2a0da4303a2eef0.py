import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: blend of Expected Improvement and UCB
    # EI is more focused on exploitation when close to optimum
    # UCB provides exploration when far from optimum
    
    # Compute Expected Improvement (for minimization)
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score (higher is better)
    beta = 2.0 + 2.0 * min(1.0, max(0.0, f_best) / 5.0)
    ucb = -mu + beta * sigma
    
    # Blend: use more EI when f_best is small (close to optimum), more UCB when f_best is large
    # Weight for EI increases as f_best approaches 0
    ei_weight = 1.0 - min(1.0, max(0.0, f_best) / 10.0)
    ucb_weight = 1.0 - ei_weight
    
    # Normalize both to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    
    scores = ei_weight * (ei / ei_max) + ucb_weight * (ucb / ucb_max)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
