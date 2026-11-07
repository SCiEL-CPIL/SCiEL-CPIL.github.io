"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard (330s wall-clock limit, use 325s as safety margin)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s kill


def _check_deadline():
    """Return True if we should stop computing (near deadline)."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = better to evaluate next)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)  # avoid division by zero
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)  # avoid division by zero
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma
    
    # For numerical stability, handle large |z|
    # When z is very negative (mu >> f_best), EI is negligible
    # When z is very positive (mu << f_best), EI ≈ delta
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = delta * cdf + sigma * pdf
    
    # Ensure EI is non-negative (numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    score = ei / cost
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
