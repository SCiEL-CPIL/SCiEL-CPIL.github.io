"""Expected Improvement acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement acquisition function for Bayesian optimization of sphere 1d.
    Returns 2000 scores, higher meaning evaluate next.
    
    EI is defined for minimization as:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Higher EI indicates a better candidate to evaluate next.
    """
    global _start_time
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback to a simple greedy strategy if we are running out of time
        return -mu

    # Ensure sigma is at least a small positive value to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the improvement
    improvement = f_best - mu
    
    # Calculate z-score
    z = improvement / sigma_safe
    
    # Calculate Expected Improvement
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is non-negative by definition (max(0, ...))
    # We return EI directly. Higher is better.
    return ei
# EVOLVE-BLOCK-END
