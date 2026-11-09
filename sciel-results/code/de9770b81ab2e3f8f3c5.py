"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses Expected Improvement (EI) as the primary acquisition function,
    which is well-suited for minimization problems with known lower bounds.
    """
    # Deadline guard: track elapsed time to ensure we return before 300s
    t0 = time.time()
    deadline = 300.0 - 2.0  # safety margin
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline before heavy computation
    if time.time() > deadline:
        # Fallback: simple EI
        improvement = f_best - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Standard Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
