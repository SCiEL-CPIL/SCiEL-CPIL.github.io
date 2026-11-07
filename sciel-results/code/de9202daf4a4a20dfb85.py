"""Acquisition function for Bayesian optimization of hartmann 3d using Knowledge Gradient."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def acquisition(mu, sigma, f_best):
    """Knowledge Gradient acquisition function.
    
    Maximizes the expected reduction in the minimum observed value.
    KG(x) = E[max(0, f_best - min(f(x), f_best))]
    where f(x) ~ N(mu(x), sigma(x)^2)
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidate points
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        KG values, higher is better
    """
    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Determine number of Monte Carlo samples based on time budget
    # We need to be fast - each call should take < 1 second
    # Use a moderate number of samples for accuracy
    n_samples = 200
    
    # Pre-compute for efficiency
    # For each sample, draw z_i ~ N(0, 1) for each candidate
    # f_i = mu_i + sigma_i * z_i
    # KG_i = E[max(0, f_best - min(f_i, f_best))]
    #      = E[max(0, f_best - f_i)] for f_i < f_best
    #      = 0 for f_i >= f_best
    
    # Efficient computation:
    # For each sample s:
    #   f_s = mu + sigma * z_s  (shape: (n_samples, n))
    #   improvement_s = max(0, f_best - f_s)  (shape: (n_samples, n))
    #   KG = mean(improvement_s, axis=0)
    
    # To be memory efficient, process in chunks
    chunk_size = min(n_samples, 50)
    kg = np.zeros(n)
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        n_chunk = end - start
        
        # Draw random samples
        z = np.random.standard_normal((n_chunk, n))
        f_samples = mu[None, :] + sigma_safe[None, :] * z
        
        # Compute improvement: max(0, f_best - f_samples)
        improvement = np.maximum(0.0, f_best - f_samples)
        
        # Accumulate
        kg += improvement.sum(axis=0)
    
    kg /= n_samples
    
    # Ensure finite values
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    return kg
# EVOLVE-BLOCK-END
