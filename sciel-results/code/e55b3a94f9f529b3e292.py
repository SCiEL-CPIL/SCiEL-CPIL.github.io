"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (POI) acquisition function.
    
    Args:
        mu: Mean of the GP posterior, shape (N,)
        sigma: Standard deviation of the GP posterior, shape (N,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Cost-aware score: We prefer high probability of improvement and low cost
    # Dividing by cost penalizes expensive points
    scores = poi / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
