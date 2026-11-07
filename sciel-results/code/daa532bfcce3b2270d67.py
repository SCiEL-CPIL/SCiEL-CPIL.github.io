import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (2000,) with posterior mean.
    sigma: array of shape (2000,) with posterior standard deviation.
    f_best: scalar, the best value observed so far.
    
    Returns:
    array of shape (2000,) with scores. Higher score means more likely to be selected.
    """
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin before 300s limit

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate improvement: how much better than f_best?
    # For minimization, improvement is f_best - mu
    improvement = f_best - mu
    
    # Standardized improvement z
    z = improvement / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF, phi is PDF of standard normal
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # If time is running out, return the current calculation (it's very fast)
    # In a more complex implementation, we might check time here, 
    # but EI is O(N) vectorized, so it finishes in milliseconds.
    
    return ei
