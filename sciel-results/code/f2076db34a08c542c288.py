import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed function value (lower is better)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores for each candidate, shape (2000,)
    """
    # UCB for minimization: we want to minimize mu - beta * sigma
    # To maximize a score, we use: beta * sigma - mu
    # We divide by cost to account for the budget constraint
    
    beta = 2.0
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute UCB score
    score = (beta * sigma_safe - mu) / cost_safe
    
    return score
