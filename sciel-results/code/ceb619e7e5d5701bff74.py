"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses a Quantile-based Thompson Sampling strategy.
    Instead of a deterministic acquisition function like UCB or EI, this method
    samples from the posterior distribution of the objective function at each
    candidate point. The score for each candidate is derived from a robust
    statistic (low quantile) of these samples, which naturally balances
    exploration (high variance regions yield lower quantiles) and exploitation
    (low mean regions yield lower quantiles).
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n = len(mu)
    if n == 0:
        return np.array([])
        
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Number of Monte Carlo samples per candidate
    # Using a moderate number to reduce variance in the score while maintaining speed
    n_samples = 20
    
    # Draw samples: shape (n_samples, n)
    # y[i, j] is the i-th sample for candidate j
    # We use standard normal random numbers scaled by sigma and shifted by mu
    # np.random.normal is fast
    z = np.random.standard_normal((n_samples, n))
    samples = mu[None, :] + sigma_safe[None, :] * z
    
    # Compute the 10th percentile of the samples for each candidate
    # A low percentile represents a "promising" low value (since we minimize)
    # If sigma is high, the 10th percentile is lower than the mean, encouraging exploration
    # If sigma is low, the 10th percentile is close to the mean, encouraging exploitation
    
    # Use np.percentile or np.quantile
    # axis=0 computes the percentile across the n_samples dimension for each candidate
    quantile_level = 10.0
    try:
        low_quantiles = np.percentile(samples, quantile_level, axis=0)
    except Exception:
        # Fallback to mean if percentile fails for some reason
        low_quantiles = np.mean(samples, axis=0)
        
    # The score should be higher for better (lower) objective values
    # So we negate the quantile
    scores = -low_quantiles
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
        
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
