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
    Cost-aware acquisition function combining adaptive LCB with EI.
    
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
        # Fast LCB fallback
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
    
    # Estimate progress: we start with 5 points, budget ~30, avg cost ~1.5
    # So ~20 total evaluations, ~15 acquisition calls
    max_expected_calls = 18
    progress = min(1.0, max(0.0, (n_calls - 1) / max_expected_calls))
    
    # Adaptive beta for LCB component
    # Early: high beta (explore), Late: low beta (exploit)
    beta_max = 2.5
    beta_min = 0.3
    beta = beta_min + (beta_max - beta_min) * (1 - progress) ** 1.5
    
    # Cost sensitivity: early low (explore freely), late high (prefer cheap)
    gamma_min = 0.5
    gamma_max = 2.0
    gamma = gamma_min + (gamma_max - gamma_min) * progress ** 2
    
    # Compute LCB for minimization: LCB = mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z) where Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    phi = norm.pdf(z_clipped)
    cdf = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * cdf + sigma_safe * phi
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Blend LCB and EI based on progress
    # Early: more LCB (exploration), Late: more EI (exploitation)
    # Use a smooth transition
    lcb_weight = 0.4 + 0.6 * (1 - progress)  # 1.0 early, 0.4 late
    ei_weight = 1.0 - lcb_weight + 0.4  # 0.4 early, 1.0 late (normalized)
    
    # Normalize both to comparable scales
    # LCB can be negative, EI is positive
    lcb_score = -lcb  # Higher LCB score = lower predicted value
    lcb_score_norm = lcb_score / (np.std(lcb_score) + 1e-9)
    ei_score_norm = ei / (np.std(ei) + 1e-9)
    
    # Combined score
    combined = lcb_weight * lcb_score_norm + ei_weight * ei_score_norm
    
    # Apply cost penalty
    cost_weighted = cost_safe ** gamma
    scores = combined / cost_weighted
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
