import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best function value observed so far.
        cost: (2000,) array of evaluation costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # UCB for minimization:
    # We want to find points where the true function value is likely to be low.
    # The lower confidence bound is LCB(x) = mu(x) - beta * sigma(x).
    # We want to minimize LCB(x).
    # Equivalently, we want to maximize -LCB(x) = beta * sigma(x) - mu(x).
    
    # We divide by cost to account for the cost-awareness.
    # Score(x) = (beta * sigma(x) - mu(x)) / cost(x)
    
    # Parameter beta controls the exploration-exploitation trade-off.
    # A fixed beta of 2.0 is a reasonable starting point for standard BO tasks.
    beta = 2.0
    
    # Ensure sigma is not zero to avoid numerical issues, though UCB doesn't divide by sigma.
    # However, if sigma is 0, the term is just -mu.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the unnormalized UCB score
    ucb_scores = beta * sigma_safe - mu
    
    # Normalize by cost
    # Cost is >= 1, so division is safe.
    scores = ucb_scores / cost
    
    # Handle any non-finite values (should not happen with valid inputs, but good practice)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
