import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    
    Uses a combination of Cost-aware Expected Improvement (cEI) and a 
    Cost-aware Uncertainty Bonus to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP surrogate
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP surrogate
    f_best : float
        Best (lowest) observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    deadline = time.time() + 280.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: ensure sigma is positive for log/cdf/pdf
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    cei = ei / cost_safe
    
    # Exploration bonus: encourage visiting high uncertainty regions
    # We scale sigma by a factor. 
    # The scale of the objective is roughly [-4, 1]. 
    # A small fraction of the objective range is a good exploration weight.
    # Let's use a fixed small weight relative to the objective scale.
    # If we don't know the scale, we can use the max sigma as a reference, 
    # but that makes the score relative to the batch.
    # Using an absolute value is more stable across iterations.
    # Hartmann 3D values are in [-4, 1]. 
    # Let's try beta = 0.1.
    beta = 0.1
    exploration_bonus = beta * sigma_safe / cost_safe
    
    # Combine
    scores = cei + exploration_bonus
    
    # Ensure finite
    scores[~np.isfinite(scores)] = 0.0
    
    # Check deadline
    if time.time() > deadline - 5.0:
        return scores
    
    return scores
