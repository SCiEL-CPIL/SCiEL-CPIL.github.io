"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for efficient minimization of Styblinski-Tang 1D.
    Uses a combination of Expected Improvement and uncertainty-weighted exploration.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    
    # EI formula for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Use a weighted combination: EI dominates, with a small exploration bonus
    # The 0.2 weight on sigma encourages exploration without overwhelming EI
    score = ei + 0.2 * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: return immediately if we're close to the limit
    elapsed = time.time() - start_time
    if elapsed > 295.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
