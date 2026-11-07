import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize deadline
    if _start_time is None:
        _start_time = time.time()
        
    # Safety margin: 330s limit, return if we are too close
    if time.time() - _start_time > 325:
        return np.zeros_like(mu)
        
    # 1. Calculate Expected Improvement (EI)
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Hard Feasibility Gate
    # Strategy: Force search to low-cost corner.
    # Calculate the 25th percentile of the costs of the 2000 candidates.
    # Only candidates with cost <= this threshold are considered "feasible".
    # This effectively ignores the most expensive 75% of candidates,
    # concentrating the search in the cheapest 25%.
    
    cost_threshold = np.percentile(cost, 25)
    
    # Mask: True if cost is within the low-cost threshold
    feasible_mask = cost <= cost_threshold
    
    # 3. Compute Score
    # Score is EI if feasible, else 0.
    # This creates a hard budget-aware gate.
    score = np.where(feasible_mask, ei, 0.0)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
