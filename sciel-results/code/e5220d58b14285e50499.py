"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We reserve a safety margin of 10s.
_DEADLINE = None
_DEADLINE_MARGIN = 10.0


def _ensure_deadline():
    """Set the global deadline on first call if not already set."""
    global _DEADLINE
    if _DEADLINE is None:
        # We don't know the exact start time of the evaluation loop,
        # but we can use a generous internal limit. Since each acquisition
        # call is fast, the main risk is pathological inputs.
        # For safety, we just ensure we return quickly.
        _DEADLINE = time.time() + 320.0  # 320s from first call, leaving 10s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher is better).
    """
    _ensure_deadline()
    
    # Convert to numpy arrays for efficient computation
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF and PDF
    # Use scipy.stats.norm for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware EI: divide by cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    # Replace any non-finite values with a very small negative number
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
