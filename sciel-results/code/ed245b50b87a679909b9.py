import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin before 300s limit


def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a strong penalty for high evaluation costs.
    
    Parameters:
    mu (np.ndarray): Mean of the Gaussian Process (2000,)
    sigma (np.ndarray): Standard deviation of the Gaussian Process (2000,)
    f_best (float): Best observed function value
    cost (np.ndarray): Cost of evaluating each candidate (2000,)
    
    Returns:
    np.ndarray: Acquisition scores (2000,)
    """
    # Deadline guard: return a safe default if we're running out of time
    if _check_deadline():
        # Return cost-based scores (prefer low cost) as a safe fallback
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / cost_safe
    
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (due to floating point errors)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost^2
    # Strong penalty for high-cost evaluations
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / (cost_safe ** 2)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
