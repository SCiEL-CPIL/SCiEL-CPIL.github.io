import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Args:
        mu: array of shape (2000,) GP mean
        sigma: array of shape (2000,) GP standard deviation
        f_best: scalar, lowest observed value
        cost: array of shape (2000,) cost of evaluating each point
    
    Returns:
        array of shape (2000,) scores (higher is better)
    """
    start_time = time.time()
    deadline = start_time + 2.0 # 2 seconds safety margin per call
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    
    # UCB parameter. 
    # A value of 2.0 is a standard exploration-exploitation trade-off.
    # Given the small budget (~15 steps), we lean towards exploration.
    beta = 2.5
    
    # Calculate UCB improvement potential
    # We want to maximize the chance of finding a value lower than f_best.
    # Lower bound on f(x) is mu - beta * sigma.
    # Expected improvement potential is f_best - (mu - beta * sigma)
    improvement_potential = f_best - mu + beta * sigma
    
    # Cost-aware score
    scores = improvement_potential / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() > deadline:
         # Return best effort if taking too long (shouldn't happen with vectorized ops)
         pass
         
    return scores
