import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Lower Confidence Bound (LCB) strategy.
    The score is (beta * sigma - mu) / cost.
    This encourages exploration (high sigma) and exploitation (low mu),
    while penalizing high-cost evaluations.
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-6)
    
    # Exploration parameter beta.
    # A value of 5.0 provides a strong exploration bias suitable for 
    # multimodal functions with a limited budget.
    beta = 5.0
    
    # Calculate the un-normalized score: we want to minimize (mu - beta * sigma),
    # so we maximize (beta * sigma - mu).
    raw_score = beta * sigma - mu
    
    # Normalize by cost to account for the cost-awareness requirement.
    # Cost is expected to be in [1, 2].
    score = raw_score / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
