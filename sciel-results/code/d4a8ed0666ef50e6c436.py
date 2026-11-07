import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Look-Ahead Expected Improvement (LAEI) approximation using UCB.
    Uses Upper Confidence Bound as a proxy for the lookahead value,
    balancing exploitation (low mu) and exploration (high sigma).
    """
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB Score for minimization: -mu + beta * sigma
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta encourages more exploration.
    beta = 1.5
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
