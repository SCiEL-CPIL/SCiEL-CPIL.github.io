"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 330  # Wall-clock limit from evaluator
_MARGIN = 10    # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _start_time > _TIMEOUT - _MARGIN:
        # If we are running out of time, return a safe, finite score
        # This shouldn't happen often as the function is fast, but it's a guard.
        # Returning zeros or a constant is safer than crashing.
        return np.zeros_like(mu)

    # Numerical stability
    epsilon = 1e-9
    sigma_safe = np.maximum(sigma, epsilon)
    cost_safe = np.maximum(cost, epsilon)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    
    # Handle extreme z values to prevent numerical issues, though scipy handles it well
    # cdf(z) -> 0 for z -> -inf, 1 for z -> inf
    # pdf(z) -> 0 for z -> +/- inf
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware score
    # Base: EI / Cost
    # Exploration Boost: Multiplicative term (1 + beta * sigma)
    # This increases the score for points with high uncertainty, 
    # encouraging exploration in the early stages or when the GP is unsure.
    # We normalize sigma slightly to prevent it from dominating if scales vary wildly,
    # but the knowledge suggests raw sigma worked well.
    
    # Let's use a dynamic beta that decreases as we approach the end?
    # We don't have access to the total cost in this function signature directly,
    # but we can infer "end" if cost is high? No, cost is per-point.
    # However, the loop condition is "total cost < 30".
    # We can't see the total cost.
    # So we use a fixed beta.
    
    beta = 0.8
    
    # The multiplicative boost
    boost = 1.0 + beta * sigma_safe
    
    score = (ei * boost) / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
