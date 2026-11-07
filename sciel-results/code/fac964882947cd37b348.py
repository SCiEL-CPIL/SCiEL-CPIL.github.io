import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted Expected Improvement (CW-EI).
    
    Args:
        mu: Array of shape (2000,) with predicted means in objective units.
        sigma: Array of shape (2000,) with predicted standard deviations in objective units.
        f_best: Scalar, the lowest observed value so far.
        cost: Array of shape (2000,) with the cost of evaluating each candidate.
        
    Returns:
        Array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-Weighted Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with very large/small values
    z = np.clip(z, -10, 10)
    
    # Compute EI components
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware: divide by cost to prioritize cheaper evaluations
    scores = EI / cost
    
    # Safety check for time
    if time.time() - t_start > 290:
        # Fallback: simple cost-weighted exploitation
        scores = np.maximum(f_best - mu, 0.0) / cost
        
    return scores
