import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) per unit cost.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviation predictions
    f_best: float, best observed value
    cost: (2000,) array of costs for each point
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    # Deadline guard
    start_time = time.time()
    deadline = 330.0
    margin = 5.0
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # UCB exploration parameter
    # Higher beta means more exploration
    beta = 2.0
    
    # Calculate UCB improvement
    # For minimization, we look for low values.
    # The lower confidence bound is mu - beta * sigma.
    # The potential improvement over f_best is f_best - (mu - beta * sigma).
    ucb_improvement = f_best - mu + beta * sigma
    
    # Cost-aware score
    score = ucb_improvement / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check time
    if time.time() > deadline - margin:
        # If we exceed time, we just return what we have.
        # The computation above is vectorized and fast, so this is a safeguard.
        pass

    return score
