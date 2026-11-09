import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,)
    """
    # UCB term for minimization: -mu + kappa * sigma
    # Cost penalty: - lambda * cost
    
    kappa = 2.0
    lambda_cost = 0.5
    
    scores = -mu + kappa * sigma - lambda_cost * cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=-1e6)
    
    return scores
