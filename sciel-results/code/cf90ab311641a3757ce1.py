import numpy as np
import time
from scipy.stats import norm

# Module-level state to track time for the deadline guard
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (cEI) acquisition function.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    global _start_time, _call_count
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
        
    _call_count += 1

    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 295.0:
        # Fallback: pick the cheapest point with lowest mean
        # Score = -cost - 0.1 * mu (prioritize cost, then exploitation)
        return -(np.asarray(cost, dtype=float) + 0.1 * np.asarray(mu, dtype=float))

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite values and valid ranges
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standardized improvement: Z = (f_best - mu) / sigma
    # We want to maximize the expected improvement per unit cost.
    # Standard EI: E[I] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    # Note: If we define improvement as f_best - y (since we minimize),
    # then I = max(0, f_best - y).
    # y ~ N(mu, sigma^2).
    # E[I] = (f_best - mu) * P(y < f_best) + E[(f_best - y) * 1(y < f_best)]
    # Let Z = (f_best - mu) / sigma.
    # P(y < f_best) = Phi(Z).
    # E[(f_best - y) * 1(y < f_best)] = sigma * phi(Z).
    # So EI = (f_best - mu) * Phi(Z) + sigma * phi(Z).
    
    Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in exp for very large positive/negative values
    # Phi(Z) is 1 for Z >> 0, 0 for Z << 0
    # phi(Z) is 0 for |Z| >> 0
    Z_clipped = np.clip(Z, -10, 10)
    
    Phi_Z = norm.cdf(Z_clipped)
    phi_Z = norm.pdf(Z_clipped)
    
    # Calculate Expected Improvement
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # Ensure EI is non-negative (numerical safety)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: EI / cost
    # Higher score means higher expected improvement per unit cost
    score = EI / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
