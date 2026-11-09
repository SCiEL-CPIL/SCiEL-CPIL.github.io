import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive parameter tuning
_state = {
    'best_f': np.inf,
    'f_history': [],
    'n_calls': 0,
    'start_time': None
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Lower Confidence Bound (LCB/UCB)
    for minimization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = prefer)
    """
    global _state
    
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['n_calls'] += 1
    _state['f_history'].append(f_best)
    
    elapsed = time.time() - _state['start_time']
    
    # Safety: if we're close to timeout, use a fast fallback
    if elapsed > 280:
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        cost_safe = np.where(cost > 1e-9, cost, 1e-9)
        # Fast UCB fallback
        beta = 1.0
        lcb = mu - beta * sigma_safe
        scores = -lcb / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Track progress
    if f_best < _state['best_f']:
        _state['best_f'] = f_best
    
    n_calls = _state['n_calls']
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Avoid division by zero
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in norm.pdf/cdf
    # norm.pdf/cdf are stable, but clipping helps
    z_clipped = np.clip(z, -10, 10)
    
    phi = norm.pdf(z_clipped)
    cdf = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * cdf + sigma_safe * phi
    
    # EI is non-negative. Higher is better.
    # Cost-aware score: EI / cost
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
