"""Bayesian optimization acquisition for Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def acquisition(mu, sigma, f_best):
    """
    Deterministic Expected Improvement (EI) acquisition function.
    """
    n = mu.shape[0]
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
