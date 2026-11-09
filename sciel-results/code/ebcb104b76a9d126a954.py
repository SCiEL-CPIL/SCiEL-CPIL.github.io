"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for the session
_SESSION_START = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Safety deadline guard
    elapsed = time.time() - _SESSION_START
    if elapsed > 280:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / np.maximum(cost, 1e-8)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost-awareness: 
    # Instead of dividing by cost directly, we use a softer penalty.
    # Cost is in [1, 2]. 
    # Using sqrt(cost) or log(cost + 1) reduces the bias against expensive regions.
    # This allows the optimizer to explore potentially better regions that are slightly more expensive.
    
    # Let's try EI / sqrt(cost)
    cost_safe = np.maximum(cost, 1e-8)
    scores = ei / np.sqrt(cost_safe)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
