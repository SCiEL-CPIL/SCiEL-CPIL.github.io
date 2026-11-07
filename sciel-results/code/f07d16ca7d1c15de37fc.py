import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    This function calculates the Expected Improvement for each candidate point
    and normalizes it by the evaluation cost to maximize efficiency.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Deadline guard
    t_start = time.time()
    deadline = 300.0
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution
    
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Calculate EI
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Maximize EI per unit cost
    # cost is >= 1.0 based on problem description
    # We divide by cost to prioritize cheaper evaluations with high expected improvement
    score = ei / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > deadline - 5.0:
        # Safety margin: return current result
        return score
    
    return score
