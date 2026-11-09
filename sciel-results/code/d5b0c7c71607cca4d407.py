"""Rewritten acquisition function using Thompson Sampling for branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Thompson Sampling: draw a sample from the posterior at each candidate
    and return the negative sample as the score (higher score = lower predicted value).
    """
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: return deterministic EI-like scores if running out of time
        sigma_safe = np.where(np.asarray(sigma, dtype=np.float64) > 1e-12, 
                              np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_safe) / sigma_safe
        from scipy.stats import norm
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Thompson Sampling: draw one sample from N(mu, sigma^2) at each candidate
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Score: negative of the sample (lower predicted value = higher score)
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
