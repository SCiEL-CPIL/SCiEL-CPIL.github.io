"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for adaptive acquisition
_state = {
    'call_count': 0,
    'f_best_history': [],
    'start_time': None
}

def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining EI with progressive exploitation.
    
    Strategy:
    - Early: exploration-heavy (higher sigma weight)
    - Late: exploitation-heavy (EI with small exploration bonus)
    - Uses time-aware deadline guard
    """
    global _state
    
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['call_count'] += 1
    _state['f_best_history'].append(f_best)
    
    # Deadline guard: 300s limit, keep 10s margin
    elapsed = time.time() - _state['start_time']
    if elapsed > 290:
        # Emergency: return pure exploitation (lowest mu)
        return -mu
    
    # Adaptive beta based on progress
    n_calls = _state['call_count']
    total_budget = 30  # total evaluations
    
    # Progress: 0 at start, 1 at end
    progress = min(1.0, n_calls / max(1, total_budget - 5))
    
    # Early: explore more (higher beta), Late: exploit more (lower beta)
    # Use a smooth transition
    beta_explore = 4.0 * (1.0 - progress) ** 2
    beta_exploit = 0.5 * progress
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Handle numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle edge cases where sigma is very small
    ei = np.where(sigma < 1e-8, np.maximum(f_best - mu, 0), ei)
    
    # Combine EI with adaptive exploration
    # Higher score = more likely to be selected
    # For minimization: high EI means good, high sigma means uncertain
    score = ei + beta_explore * sigma - beta_exploit * (f_best - mu)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
