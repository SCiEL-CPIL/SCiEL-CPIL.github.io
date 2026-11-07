import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Computes cost-aware Upper Confidence Bound scores.
    
    Args:
        mu: Array of shape (2000,) containing mean predictions.
        sigma: Array of shape (2000,) containing standard deviations.
        f_best: Scalar or array containing the best observed value.
        cost: Array of shape (2000,) containing the cost of evaluation.
        
    Returns:
        Array of shape (2000,) with acquisition scores.
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # UCB for minimization: We want to maximize the probability of finding a low value.
    # The "upper" confidence bound for the minimum is actually the lower bound of the uncertainty.
    # However, standard UCB for maximization is mu + beta*sigma.
    # For minimization, we can maximize -mu + beta*sigma.
    # This balances exploitation (low mu) and exploration (high sigma).
    
    beta = 2.0
    
    # Calculate UCB score
    ucb = -mu + beta * sigma
    
    # Cost-aware adjustment: Divide by cost
    scores = ucb / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
