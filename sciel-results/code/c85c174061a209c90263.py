"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300  # 300 seconds limit per evaluation, keep safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PoI).
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Fallback: return zeros or a simple heuristic if we're running out of time
        # But we must return finite scores.
        # Since we can't compute, return a constant that allows the loop to finish or pick randomly.
        # However, the task says "anything else invalidates". So we must return finite numbers.
        # A safe fallback is to return 0 for all, which might lead to random picks if ties are broken randomly,
        # or we can return 1/cost to at least prefer cheap points.
        return 1.0 / (cost + 1e-9)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # PoI is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Cost-aware: PoI per unit cost
    # We want to maximize this
    scores = poi / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
