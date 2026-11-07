import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    if time.time() > _deadline:
        return np.ones(len(mu))

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If sigma is very small and mu < f_best, z is large positive.
    # Phi(z) ~ 1, phi(z) ~ 0. EI ~ f_best - mu.
    # If sigma is very small and mu > f_best, z is large negative.
    # Phi(z) ~ 0, phi(z) ~ 0. EI ~ 0.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical errors might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI divided by cost
    # Higher EI per unit cost is better
    scores = ei / (cost + eps)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
