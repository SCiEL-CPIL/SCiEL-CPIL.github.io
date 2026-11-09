import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we finish well within the 300s limit.
    # The computation is vectorized and fast, but we add a safety check.
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for the standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: divide by cost
    # This encourages evaluating points that are both promising (high EI) and cheap
    scores = ei / cost
    
    # Handle any non-finite values to ensure validity
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time - if we're close to the deadline, return immediately
    # Since the computation is already done, this is just a safety guard
    if time.time() - start_time > 290:
        return scores
        
    return scores
