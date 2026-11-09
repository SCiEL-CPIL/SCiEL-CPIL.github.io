"""Combined acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track call count and detect new seeds
_state = {
    'n_calls': 0,
    'prev_f_best': None,
    'start_time': time.time()
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a robust 
    deadline guard and state management.
    
    Parameters:
    - mu: array of shape (2000,) with posterior means
    - sigma: array of shape (2000,) with posterior standard deviations
    - f_best: scalar, the best observed value so far
    
    Returns:
    - array of shape (2000,) with acquisition scores (higher = better)
    """
    # Convert to numpy arrays for efficient vectorized operations
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Detect if a new seed/run has started
    # f_best typically decreases. If it increases significantly, it's likely a new run.
    if _state['prev_f_best'] is not None and f_best > _state['prev_f_best'] + 1e-6:
        _state['n_calls'] = 0
        _state['start_time'] = time.time()
        
    _state['prev_f_best'] = f_best
    n = _state['n_calls']
    _state['n_calls'] = n + 1
    
    # Deadline Guard
    # The total budget is 330s. If we are close to the limit, return quickly.
    elapsed = time.time() - _state['start_time']
    if elapsed > 325.0:
        # Fallback to pure EI to ensure we finish
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute Expected Improvement (EI)
    # This is the proven baseline (Inspiration 1, score 1.0).
    # Previous attempts to blend with UCB (score 0.29) were harmful, likely due to
    # excessive exploration or state management issues.
    # We stick to pure EI for robustness.
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all values are finite
    score = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
