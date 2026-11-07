"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_deadline_start = time.time()
_DEADLINE_LIMIT = 320.0  # safety margin before 330s kill


def _check_deadline():
    """Return True if we should abort and return a safe default."""
    return (time.time() - _deadline_start) > _DEADLINE_LIMIT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Cost-Aware Expected Improvement: EI / sqrt(cost)
    This directly optimizes improvement per unit cost, naturally favoring
    high-improvement points in low-cost regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    if _check_deadline():
        return np.zeros(len(mu))
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be by definition, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: divide by sqrt(cost)
    # This is a convex penalty that naturally trades off improvement vs cost
    cost_penalty = np.sqrt(cost_safe)
    
    # Primary score: EI / sqrt(cost)
    scores = ei / cost_penalty
    
    # Add a small exploration component to avoid premature convergence
    # Use posterior standard deviation normalized by cost
    exploration = 0.01 * sigma_safe / cost_safe
    
    scores = scores + exploration
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
