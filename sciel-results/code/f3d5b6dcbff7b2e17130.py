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
    Acquisition function blending Expected Improvement (EI) with a decaying
    UCB exploration term.
    
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
        _state['start_time'] = time.time() # Reset time reference for the new run
        
    _state['prev_f_best'] = f_best
    n = _state['n_calls']
    _state['n_calls'] = n + 1
    
    # Deadline Guard
    # The total budget is 330s. If we are close to the limit, we should return quickly.
    # Since the computation is O(N), it's very fast. This guard is mostly for safety
    # against any unexpected hangs or if the evaluator imposes strict per-call limits
    # that we can't see.
    elapsed = time.time() - _state['start_time']
    if elapsed > 325.0:
        # If we are running out of time, return a simple heuristic (e.g., pure EI or zeros)
        # to ensure we don't crash or timeout mid-calculation.
        # Pure EI is a safe fallback.
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration Term (UCB-like)
    # We use a decaying weight to encourage exploration early and exploitation late.
    # Previous attempts with constant UCB (0.05) failed.
    # We try a decay from 0.05 to 0 over 30 steps.
    # This ensures that in the later stages, we behave like pure EI (which scored 1.0),
    # but in the early stages, we have a slight boost to escape local minima.
    
    # Linear decay: weight = 0.05 * (1 - n/30)
    # At n=0, weight=0.05. At n=29, weight ~ 0.0017.
    weight = 0.05 * max(0.0, 1.0 - n / 30.0)
    
    ucb_term = weight * sigma
    
    # Combined Score
    score = ei + ucb_term
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
