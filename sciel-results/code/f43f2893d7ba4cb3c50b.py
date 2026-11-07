import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE_MARGIN = 15.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        return np.full(2000, 0.0)
    
    # Convert to numpy arrays and ensure finiteness
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe sigma for division
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    # Compute Z-score: (f_best - mu) / sigma
    # Note: For minimization, we want mu < f_best.
    # Z > 0 implies mu < f_best (potential improvement)
    Z = (f_best - mu) / safe_sigma
    
    # Compute standard normal CDF and PDF
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # Numerical safety: EI should be >= 0
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score
    # We want high EI and low cost.
    score = EI / cost
    
    # Final safety check
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
