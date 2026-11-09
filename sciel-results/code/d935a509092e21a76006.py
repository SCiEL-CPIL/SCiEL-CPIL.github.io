import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        # Fallback: Simple UCB or random? 
        # If we are out of time, we should return something valid.
        # A safe fallback is to maximize -mu / cost (greedy)
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
    
    # Ensure sigma is positive for EI calculation
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Handle division by zero or very small sigma
    # If sigma is tiny, Z is huge. 
    # We can compute Z safely.
    Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in norm.cdf/pdf for extreme values
    # norm.cdf(-10) ~ 0, norm.pdf(-10) ~ 0
    # norm.cdf(10) ~ 1, norm.pdf(10) ~ 0
    # This is generally stable in scipy, but let's be safe.
    Z_clipped = np.clip(Z, -10, 10)
    
    # Calculate EI
    # Note: If f_best - mu is negative (mu > f_best), EI should be 0 or very small.
    # The formula handles this: if Z < 0, (f_best - mu) is negative, Phi(Z) is small, 
    # but the term is positive? 
    # Let's check: 
    # Let d = f_best - mu.
    # If d < 0, Z = d/sigma < 0.
    # EI = d * Phi(Z) + sigma * phi(Z).
    # For d < 0, EI is actually negative in the raw formula? 
    # No, EI is defined as E[max(0, f_best - f(x))]. It is always non-negative.
    # The formula (f_best - mu)Phi(Z) + sigma phi(Z) yields negative values when Z is very negative?
    # Let's test Z = -10. Phi(-10) ~ 0. phi(-10) ~ 0. Result ~ 0.
    # Let's test Z = -1. d = -sigma.
    # EI = -sigma * Phi(-1) + sigma * phi(-1) = sigma * (phi(-1) - Phi(-1)).
    # phi(-1) ~ 0.24, Phi(-1) ~ 0.15. Result is positive.
    # So the formula is correct and non-negative.
    
    EI = (f_best - mu) * norm.cdf(Z_clipped) + sigma * norm.pdf(Z_clipped)
    
    # Ensure non-negative
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: EI / Cost
    # This prioritizes improvements per unit cost.
    scores = EI / cost
    
    # Final sanitization
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
