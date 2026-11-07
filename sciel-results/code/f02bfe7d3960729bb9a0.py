# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) for minimization.
    Score is (beta * sigma - mu) / cost.
    This encourages exploration (high sigma) and exploitation (low mu), normalized by cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Finite acquisition scores (higher = more desirable).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no division by zero
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB parameter (exploration/exploitation trade-off)
    # For minimization, we want to minimize mu - beta * sigma.
    # So we maximize beta * sigma - mu.
    beta = 2.0
    
    # Calculate UCB score
    ucb_value = beta * sigma - mu
    
    # Normalize by cost
    scores = ucb_value / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
