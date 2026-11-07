import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300.0 # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are dangerously close to the limit, return a safe fallback
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    # Safe sigma to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)] where f ~ N(mu, sigma)
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi + safe_sigma * phi
    
    # Cost-aware score: EI / cost
    safe_cost = np.maximum(cost, 1e-9)
    score = ei / safe_cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
