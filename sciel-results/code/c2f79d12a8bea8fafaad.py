import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    mu: array of mean predictions from GP
    sigma: array of standard deviations from GP
    f_best: best observed value so far
    cost: array of costs for each candidate
    
    Returns:
    scores: array of scores (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero or invalid std
    sigma = np.maximum(sigma, 1e-8)
    
    # Sample from the posterior distribution for each candidate
    # f(x) ~ N(mu, sigma^2)
    samples = np.random.normal(mu, sigma)
    
    # We want to minimize f(x).
    # The "reward" for a candidate is -f(x).
    # We divide by cost to account for the cost-awareness.
    # Higher score indicates a better candidate to evaluate next.
    scores = -samples / cost
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=0.0, neginf=-np.inf)
    
    return scores
