import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB per unit cost.
    
    Args:
        mu: Mean of the GP posterior (shape 2000,)
        sigma: Std dev of the GP posterior (shape 2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Acquisition scores (shape 2000,), higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.clip(sigma, 1e-9, 1e6)
    cost = np.clip(cost, 1e-9, 1e6)
    
    # UCB exploration parameter
    # Chosen based on typical BO budget (~20-30 evals)
    beta = 2.5
    
    # UCB for minimization: mu - beta * sigma
    # We want to minimize this value.
    ucb = mu - beta * sigma
    
    # Score: Higher is better.
    # We want to select the point with the lowest UCB, penalized by cost.
    # score = -UCB / cost
    score = -ucb / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
