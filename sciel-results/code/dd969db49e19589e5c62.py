"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Probability of Improvement (PoI) with cost normalization.
    
    PoI is more stable under varying sigma than EI and naturally couples
    exploration with cost efficiency. We use PoI/cost as the acquisition score.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-10)
        return 1.0 / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate z-score for PoI
    # PoI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Handle numerical issues in z
    z = np.where(np.isfinite(z), z, 0.0)
    
    # Calculate Phi(z) - the probability of improvement
    poi = norm.cdf(z)
    
    # Ensure PoI is in [0, 1] and finite
    poi = np.clip(poi, 0.0, 1.0)
    poi = np.where(np.isfinite(poi), poi, 0.0)
    
    # Cost-aware score: PoI / cost
    # This encourages finding improvements at low cost
    scores = poi / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
