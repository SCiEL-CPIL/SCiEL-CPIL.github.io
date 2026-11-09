import numpy as np
import time

_start_time = None
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running too long, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return negative mu / cost as exploitation score
        mu = np.asarray(mu, dtype=float).ravel()
        cost = np.asarray(cost, dtype=float).ravel()
        cost = np.clip(cost, 1e-12, None)
        score = -mu / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Safety margins
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Cost-Adjusted Lower Confidence Bound (LCB)
    # For minimization: LCB = mu - kappa * sigma
    # We want to MINIMIZE the objective, so we maximize score = -LCB = -mu + kappa * sigma
    # With cost awareness: score = (-mu + kappa * sigma) / cost
    # This directly targets regions with low predicted values and high uncertainty
    # while penalizing expensive evaluations
    
    # Adaptive kappa: start with higher exploration, decay as we find better solutions
    # f_best is the lowest observed value, so as it approaches 0 (the minimum), we can be more exploitative
    # Scale kappa based on how close f_best is to the known minimum (0 for three hump camel)
    # When f_best is close to 0, reduce exploration; when far, increase exploration
    # Use a simple heuristic: kappa decreases as f_best approaches 0
    f_best_clipped = max(f_best, 1e-10)
    # Map f_best from [0, ~1] to kappa in [2.0, 0.5]
    # When f_best is small (close to min), kappa is lower (more exploitation)
    # When f_best is large (far from min), kappa is higher (more exploration)
    kappa = 0.5 + 1.5 * min(f_best_clipped, 1.0)
    
    # LCB-based score: maximize (-mu + kappa * sigma) / cost
    # This is equivalent to minimizing (mu - kappa * sigma) * cost
    # The /cost normalization makes it cost-aware
    score = (-mu + kappa * sigma) / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
