import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: 
    # - When f_best is far from optimum (large), we need more exploration (higher beta).
    # - As f_best approaches 0, we should exploit more (lower beta).
    # Weierstrass 1D on [-0.5, 0.5] has min 0 and max around 0.5-1.0.
    # Let's assume a scale of 1.0 for normalization.
    f_scale = 1.0
    f_norm = np.clip(f_best / f_scale, 0.0, 1.0)
    
    # Beta starts high (3.0) when f_best is high, and drops to 0.5 when f_best is near 0.
    beta = 0.5 + 2.5 * f_norm
    
    # LCB for minimization: mu - beta * sigma
    # We want higher scores to be evaluated.
    # So we take the negative: -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
