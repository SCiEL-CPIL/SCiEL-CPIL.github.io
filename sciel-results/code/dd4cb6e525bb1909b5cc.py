import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound (LCB) for exploration
    # LCB = mu - kappa * sigma
    # For minimization, we want low values, so we pick points with low mu - kappa*sigma
    # But acquisition should return higher scores for better candidates
    # So we negate: score = -LCB = -mu + kappa*sigma
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive kappa based on uncertainty
    # Start with moderate exploration, can be tuned
    kappa = 2.0
    
    # LCB acquisition (negated so higher is better)
    score = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
