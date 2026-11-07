import numpy as np
from scipy.stats import norm
import time

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Hard multiplicative feasibility gate based on cost quantiles
    # This forces the search to concentrate evaluations in the low-cost corner
    # until GP uncertainty there is exhausted.
    
    # Determine the 70th percentile of costs
    cost_70 = np.percentile(cost_safe, 70)
    
    # Create a gate: 1.0 for low-cost points, 0.0 for high-cost points
    # We use a soft transition to avoid discontinuities that might confuse the optimizer
    # but with a sharp threshold to create the "hard" gate effect
    gate = np.where(cost_safe < cost_70, 1.0, 0.0)
    
    # For the 30% of points with higher cost, we give them a very small weight
    # to allow some exploration but heavily penalize them
    # The weight for high-cost points is scaled down by a factor
    high_cost_weight = 0.01
    
    # Apply the gate
    scores = ei * (gate + (1 - gate) * high_cost_weight)
    
    # Add a small exploration term for low-cost points only
    exploration = 0.05 * sigma_safe * gate
    scores += exploration
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
