import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Quantile-based Expected Improvement (QEI) targeting the 95th percentile
    of the improvement distribution. This is equivalent to maximizing the 5th 
    percentile (lower bound) of the posterior predictive distribution of the 
    function value. This approach provides robust exploration in multimodal 
    landscapes by prioritizing regions with high-confidence potential for 
    improvement over regions with only high mean potential.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate the 95th percentile of the improvement distribution.
    # Improvement I = f_best - Y, where Y ~ N(mu, sigma^2).
    # I ~ N(f_best - mu, sigma^2).
    # The 95th percentile of I is: (f_best - mu) + z_0.95 * sigma
    # z_0.95 is the 95th percentile of the standard normal distribution.
    z_95 = norm.ppf(0.95)  # Approximately 1.64485
    
    # Score is the quantile of the improvement.
    # Higher score means higher confidence in a large improvement.
    score = (f_best - mu) + z_95 * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Time guard: If we are taking too long, return a simple exploitation-focused score
    if time.time() - start_time > 290:
        return -mu
    
    return score
