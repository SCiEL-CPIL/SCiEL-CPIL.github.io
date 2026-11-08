import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: Array of predicted means from the GP, shape (N,).
        sigma: Array of predicted standard deviations from the GP, shape (N,).
        f_best: The best observed function value (scalar).
        cost: Array of costs for each candidate, shape (N,).
        
    Returns:
        Array of acquisition scores, shape (N,). Higher is better.
    """
    # Exploration-exploitation trade-off parameter
    beta = 2.5
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # UCB score: We want to minimize mu - beta * sigma
    # So we maximize beta * sigma - mu
    # Divide by cost to prefer cheaper points
    scores = (beta * sigma - mu) / cost
    
    # Replace non-finite values with a very low score to ensure they are never picked
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
