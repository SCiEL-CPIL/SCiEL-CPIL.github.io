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
    
    # EI Calculation
    # f_best is a scalar, mu and sigma are arrays
    delta = f_best - mu
    z = delta / safe_sigma
    
    # To improve numerical stability for large negative/positive z
    # norm.cdf and norm.pdf handle this reasonably well in scipy
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = delta * cdf_z + safe_sigma * pdf_z
    
    # Cost-aware score
    # Use power 1.0 (linear) instead of 1.5 to reduce penalty on cost
    # This encourages exploring higher cost points more if they have high EI
    safe_cost = np.maximum(cost, 1e-9)
    score = ei / safe_cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
