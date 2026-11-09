import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: numpy array of shape (N,) with posterior means.
        sigma: numpy array of shape (N,) with posterior standard deviations.
        f_best: float, the best observed value so far.
        
    Returns:
        numpy array of shape (N,) with scores. Higher is better.
    """
    # UCB for minimization: maximize -mu + beta * sigma
    # beta controls the exploration-exploitation trade-off.
    # A value around 2-4 is typical for low-budget BO on 2D functions.
    beta = 3.0
    
    # Calculate scores
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite to satisfy the contract
    # If any values are NaN or Inf, replace them with a very low score (or 0, but lower is better for "don't pick")
    # Actually, the contract says "Return 2000 finite scores... anything else makes the whole result invalid".
    # So we must ensure no NaN/Inf.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
