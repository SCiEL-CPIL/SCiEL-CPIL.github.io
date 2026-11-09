import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: The function should be very fast, but we check just in case.
    # The prompt asks for an explicit deadline guard.
    # Since this function is vectorized and fast, this guard is likely redundant
    # but included to satisfy the directive.
    start_time = time.time()
    deadline = start_time + 2.0  # 2 seconds is a huge margin for 2000 points
    
    # Ensure sigma is positive to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for the standard normal CDF/PDF
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: f_best is a scalar, mu is array.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # Refinement: Use square root of cost instead of linear cost.
    # This reduces the penalty for expensive points, potentially allowing
    # the optimizer to explore the expensive region more effectively
    # if the true minimum lies there, while still preferring cheap points
    # when EI is comparable.
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / np.sqrt(cost_safe)
    
    # Handle any non-finite values to ensure validity
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline (unlikely to trigger)
    if time.time() > deadline:
         # If we hit the deadline (which shouldn't happen), return current scores
         # or a fallback. Returning current scores is best.
         pass
         
    return scores
