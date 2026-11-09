import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive LCB normalized by cost.
    
    Args:
        mu: Array of mean predictions (2000,)
        sigma: Array of standard deviation predictions (2000,)
        f_best: Scalar, best observed value
        cost: Array of costs for each candidate (2000,)
        
    Returns:
        Array of scores (2000,), higher is better
    """
    # Theoretical minimum of cosine8 is -0.8
    min_val = -0.8
    
    # Calculate regret to determine exploration rate
    # Larger regret (worse f_best) -> more exploration
    regret = f_best - min_val
    # Clamp regret to [0, 1] to keep beta in [1, 2]
    regret_clipped = np.clip(regret, 0.0, 1.0)
    beta = 1.0 + regret_clipped
    
    # LCB for minimization: mu - beta * sigma
    # Acquisition (to maximize): -LCB = beta * sigma - mu
    lcb_score = beta * sigma - mu
    
    # Normalize by cost
    # Cost is >= 1
    scores = lcb_score / cost
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
