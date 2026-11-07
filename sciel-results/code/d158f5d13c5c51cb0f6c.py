import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    Args:
        mu: Array of GP means (2000,)
        sigma: Array of GP standard deviations (2000,)
        f_best: Scalar, lowest observed function value
        cost: Array of evaluation costs (2000,)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for PI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Soft cost penalty: penalize high cost by dividing by sqrt(cost)
    # This is softer than dividing by cost, allowing exploration in moderately expensive areas
    score = pi / np.sqrt(cost)
    
    # Ensure output is finite
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return score
