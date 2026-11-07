import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    sigma = np.maximum(sigma, 1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Thompson Sampling
    # Draw a sample from the posterior
    z = np.random.normal(0, 1, size=mu.shape)
    f_sample = mu + sigma * z
    
    # We want to minimize the function value, so we maximize the negative value.
    # Cost-aware: divide by cost to prefer cheaper evaluations.
    # Note: cost is in [1, 2].
    scores = -f_sample / cost
    
    # Sanitize output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
