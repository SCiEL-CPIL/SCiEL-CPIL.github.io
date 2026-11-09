# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s safety margin before 300s limit


def _check_deadline():
    """Return True if we should stop computing to stay within deadline."""
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI) per unit cost.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points (in objective units)
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points (in objective units)
    f_best : float
        Lowest observed objective value
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next)
    """
    # Handle edge cases
    if _check_deadline():
        # Fallback: simple expected improvement per cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / np.maximum(cost, 1e-9)
    
    # Use Probability of Improvement (PoI) which is more robust to high-variance regions
    # PoI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Add a small exploration term to avoid completely ignoring uncertain regions
    # This helps navigate the narrow Rosenbrock valley
    # Use a weighted combination: primarily PoI, with a small EI component
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine PoI and EI, weighted toward PoI for stability
    # The weight on EI is small to avoid being dominated by high-sigma regions
    score = 0.8 * poi + 0.2 * ei / np.maximum(np.abs(f_best) + 1.0, 1.0)
    
    # Normalize by cost (cost-aware)
    cost_safe = np.maximum(cost, 1e-9)
    scores = score / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
