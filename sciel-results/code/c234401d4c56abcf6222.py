import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check if we're approaching the deadline (330s limit, use 325s as safety margin)
    # If time is running out, return a fast, simple score to avoid timeout
    elapsed = time.time() - _start_time
    if elapsed > 325:
        safe_cost = np.maximum(cost, 1e-10)
        # Simple fallback: prioritize low mu and low cost
        return -mu / safe_cost

    # Small epsilon to prevent division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # This is a standard robustness improvement
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate EI components
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, if mu < f_best, z is positive, Phi(z) is close to 1.
    # If mu > f_best, z is negative, Phi(z) is small.
    ei = (f_best - mu) * cdf_val + safe_sigma * pdf_val
    
    # Cost-Aware Score: EI per unit cost
    # This directly optimizes for improvement efficiency relative to the evaluation cost
    score = ei / safe_cost
    
    # Add a small exploration bonus to encourage diversity in early stages
    # This is refined from the high-performing inspiration
    exploration_bonus = 0.01 * safe_sigma / safe_cost
    
    score = score + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
