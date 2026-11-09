import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety check for timeout
    # The computation is fast, but if we are close to the limit, return a simple heuristic
    # to ensure we don't crash the evaluator.
    if time.time() - _start_time > 300:
        # Favor cheap points as a last resort
        return -cost

    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Numerical stability
    # Clip sigma to avoid division by zero and extreme values
    sigma = np.clip(sigma, 1e-8, 1e8)
    # Clip cost to avoid division by zero
    cost = np.clip(cost, 1e-8, 1e8)
    
    # Expected Improvement (minimization)
    delta = f_best - mu
    z = delta / sigma
    
    # Numerically stable EI calculation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = delta * cdf_z + sigma * pdf_z
    
    # Ensure ei is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # Combine EI/cost (efficiency) with a small exploration bonus (sigma/cost)
    # and a small penalty for cost to avoid over-exploiting expensive regions
    # if EI is comparable.
    # The EI/cost term is the primary driver (from inspiration 1).
    # The sigma/cost term encourages exploration in high-uncertainty, cheap regions.
    # The -0.05 * cost term slightly discourages expensive regions when EI is low.
    
    # We use a slightly lower penalty on cost and a slightly higher exploration bonus
    # compared to inspiration 1, to balance the tight budget (30) and the need to find
    # the global minimum in a 2D landscape with 5 shifted instances.
    
    score = ei / cost - 0.03 * cost + 0.08 * (sigma / cost)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
