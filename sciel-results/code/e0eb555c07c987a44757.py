import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (POI) acquisition function.
    
    Parameters:
    mu: array of shape (N,), predicted mean from GP.
    sigma: array of shape (N,), predicted standard deviation from GP.
    f_best: float, best observed function value so far.
    cost: array of shape (N,), cost of evaluating each candidate.
    
    Returns:
    scores: array of shape (N,), acquisition scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Cost-aware score: POI / cost
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    scores = poi / cost_safe
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
