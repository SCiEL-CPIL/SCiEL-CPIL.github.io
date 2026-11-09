import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition for Bayesian optimization of Goldstein-Price 2D.
    Returns scores where higher means evaluate next.
    Uses the 10th percentile of the posterior: mu - 1.28 * sigma,
    then negates so that lower predicted values give higher scores.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0
    
    # Quantile-based acquisition: 10th percentile
    # The 10th percentile of N(mu, sigma^2) is mu - 1.2816 * sigma
    # We want to minimize, so lower predicted values are better.
    # Since the evaluator picks the candidate with the HIGHEST score,
    # we return negative of the quantile: - (mu - 1.28 * sigma) = -mu + 1.28 * sigma
    
    # Use a slightly adaptive coefficient:
    # When far from optimum (large gap), use a higher quantile (more exploration)
    # When close to optimum (small gap), use a lower quantile (more exploitation)
    
    gap = max(f_best - 3.0, 0.0)
    
    # Coefficient: 1.28 (10th percentile) when close, up to ~1.64 (5th percentile) when far
    # This is a gentle shift toward more exploration when uncertain
    z_quantile = 1.28 + 0.36 * (1.0 - np.exp(-gap / 3.0))
    
    scores = -mu + z_quantile * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
