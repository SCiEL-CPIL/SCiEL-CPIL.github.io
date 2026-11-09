import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    Samples from the posterior predictive distribution and picks the minimum.
    Since we return scores for the evaluator to pick the max, we return the negative of the sample.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Generate random samples from standard normal
    # Using np.random.randn for efficiency
    z = np.random.randn(len(mu))
    
    # Posterior predictive samples
    samples = mu + sigma * z
    
    # We want to minimize the function value, so we maximize the negative sample
    score = -samples
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() - start_time > 290:
        return -mu
    
    return score
