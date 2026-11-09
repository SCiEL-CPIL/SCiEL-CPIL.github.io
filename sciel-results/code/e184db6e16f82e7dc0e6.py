import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: Mean of the GP posterior, shape (N,)
        sigma: Standard deviation of the GP posterior, shape (N,)
        f_best: The best observed function value so far.
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores are preferred.
    """
    # Beta controls the exploration-exploitation trade-off.
    # Higher beta encourages exploration of high-uncertainty regions.
    beta = 2.0
    
    # Ensure sigma is positive to avoid numerical issues, though it should be > 0
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # For minimization, we want to maximize -mu + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma (the lower bound)
    scores = -mu + beta * sigma_safe
    
    return scores
