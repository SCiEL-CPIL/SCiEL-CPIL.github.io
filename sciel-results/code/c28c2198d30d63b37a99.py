"""
Strategy: Multi-sample Monte Carlo Expected Improvement (MC-EI)
This approach differs from the current Thompson Sampling (single sample) by
computing the expected value of the improvement over f_best for each candidate
point, averaged over multiple Monte Carlo samples. This reduces the variance of
the acquisition scores compared to single-sample TS, leading to more stable
selection of points with high expected improvement, which is crucial for
escaping local optima in multi-modal landscapes like Styblinski-Tang.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Multi-sample Monte Carlo Expected Improvement.
    
    Args:
        mu: array of shape (n,), posterior mean
        sigma: array of shape (n,), posterior std
        f_best: float, best observed value so far
    
    Returns:
        scores: array of shape (n,), higher is better
    """
    n = len(mu)
    
    # Safety check for time
    if time.time() > _DEADLINE - 5.0:
        # Fallback to analytic EI if running out of time
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Number of MC samples for EI estimation
    # 20 samples is a good balance: reduces variance vs 1 (TS) but is fast.
    # For n=2000, 20x2000 = 40,000 ops, which is trivial for NumPy.
    n_samples = 20
    
    # Generate standard normal samples
    # Shape: (n_samples, n)
    Z = np.random.standard_normal((n_samples, n))
    
    # Simulate true values y ~ N(mu, sigma^2)
    # y[s, i] = mu[i] + sigma[i] * Z[s, i]
    y = mu[None, :] + sigma_safe[None, :] * Z
    
    # Compute improvement for each sample and candidate
    # Improvement = max(0, f_best - y)
    # Shape: (n_samples, n)
    improvements = np.maximum(0, f_best - y)
    
    # Average improvement over samples
    # Shape: (n,)
    scores = np.mean(improvements, axis=0)
    
    # Ensure finite
    scores[~np.isfinite(scores)] = -1e9
    
    return scores
# EVOLVE-BLOCK-END
