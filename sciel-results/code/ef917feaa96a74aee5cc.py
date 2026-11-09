"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track state across calls
_state = {
    'start_time': None,
    'call_count': 0,
    'f_best_history': []
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement with a small exploration bonus.
    
    Uses EI as the primary objective (exploitation) plus a small weighted
    sigma term to encourage exploration of uncertain regions. This simple
    approach has been shown to work well for 6D Hartmann optimization.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    # Initialize state on first call
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['call_count'] += 1
    
    # Deadline guard: 330s wall-clock limit, use 325s as safety margin
    elapsed = time.time() - _state['start_time']
    if elapsed > 325:
        # Return EI-only scores quickly to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    n = len(mu)
    
    # Track f_best history
    _state['f_best_history'].append(f_best)
    
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus based on predictive uncertainty
    # The 0.1 weight balances exploitation and exploration
    scores = ei + 0.1 * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
