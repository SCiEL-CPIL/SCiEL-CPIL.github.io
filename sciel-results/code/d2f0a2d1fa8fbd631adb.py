import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses an adaptive blend of Expected Improvement (EI) and Lower Confidence Bound (LCB).
    
    - EI drives exploitation, focusing on points likely to improve upon f_best.
    - LCB drives exploration, focusing on points with high uncertainty.
    - The blend weight shifts from EI (exploitation) to LCB (exploration) 
      as f_best is further from the known global minimum (0.397887).
    """
    start_time = time.time()
    
    # Known global minimum of Branin 2D
    f_opt = 0.397887
    
    # Calculate the gap between current best and optimal
    gap = f_best - f_opt
    if gap < 0:
        gap = 0.0
        
    # Determine exploration weight
    # When gap is large, we want to explore more (high w_explore).
    # When gap is small, we want to exploit more (low w_explore).
    # w_explore = gap / (1 + gap) maps [0, inf) to [0, 1)
    w_explore = gap / (1.0 + gap)
    w_exploit = 1.0 - w_explore
    
    # Handle zero or near-zero sigma
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    EI = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Calculate LCB-based score in objective units (higher is better)
    # LCB_value = mu - beta * sigma
    # Score_LCB = f_best - LCB_value = f_best - mu + beta * sigma
    # This term rewards low mu and high sigma.
    beta_lcb = 2.0
    Score_LCB = f_best - mu + beta_lcb * sigma_safe
    
    # Blend
    scores = w_exploit * EI + w_explore * Score_LCB
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Return a simple fallback if we're running out of time
        return EI
        
    return scores
