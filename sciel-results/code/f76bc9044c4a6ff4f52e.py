import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    Args:
        mu: Array of mean predictions (shape 2000).
        sigma: Array of standard deviations (shape 2000).
        f_best: Best observed value so far.
        cost: Array of costs for each candidate (shape 2000).
        
    Returns:
        Array of scores (shape 2000). Higher scores indicate better candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Weight by inverse cost to favor cheaper evaluations
    # Cost is in range [1, 2], so 1/cost is in [0.5, 1]
    scores = pi / cost
    
    # Handle any potential numerical issues (NaN, Inf)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
