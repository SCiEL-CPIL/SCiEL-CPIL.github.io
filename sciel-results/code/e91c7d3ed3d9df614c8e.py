import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0
_rng = np.random.default_rng(42)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    Uses Thompson Sampling with cost normalization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count, _rng
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return exploit-heavy fallback
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, 1e6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample from GP posterior
    # For minimization, we want low values
    # Sample: mu - sigma * z where z ~ N(0,1)
    z_samples = _rng.standard_normal(size=mu.shape)
    ts_samples = mu - sigma * z_samples
    
    # Cost-aware Thompson Sampling score
    # Higher is better: we want low ts_samples (good function values)
    # and low cost
    ts_score = -ts_samples / cost_safe
    
    # Also compute EI as a complementary signal
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    cost_aware_ei = ei / cost_safe
    
    # Blend Thompson Sampling and EI based on call count
    # Early: more TS for exploration, Later: more EI for exploitation
    if _call_count <= 3:
        ts_weight = 0.7
    elif _call_count <= 8:
        ts_weight = 0.5
    elif _call_count <= 15:
        ts_weight = 0.3
    elif _call_count <= 30:
        ts_weight = 0.15
    else:
        ts_weight = 0.05
    
    ei_weight = 1.0 - ts_weight
    
    scores = ts_weight * ts_score + ei_weight * cost_aware_ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
