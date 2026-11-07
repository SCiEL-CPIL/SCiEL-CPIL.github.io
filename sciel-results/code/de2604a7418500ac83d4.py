"""Cost-aware acquisition using Probability of Improvement (POI) divided by cost."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level persistent state across calls within one evaluation run
_STATE = {
    "start_time": None,
}


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement (POI) / cost.
    
    POI(x) = Phi((f_best - mu(x)) / sigma(x))
    Score(x) = POI(x) / cost(x)
    
    This prioritizes points with a high likelihood of beating the current best
    at the lowest possible cost. Unlike EI, it ignores the magnitude of improvement
    in favor of certainty, and unlike LCB, it strictly enforces a cost-efficiency ratio.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if _STATE["start_time"] is None:
        _STATE["start_time"] = time.time()
    
    elapsed = time.time() - _STATE["start_time"]
    if elapsed > 290.0:  # Safety margin before 330s kill
        # Fallback: simple POI/cost with minimal computation
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        cost_safe = np.maximum(cost, 1e-10)
        scores = poi / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma close to 0
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate Probability of Improvement
    # POI = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Cost-aware score: POI / cost
    scores = poi / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
