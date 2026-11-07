# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 300s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of expected improvement and uncertainty,
    normalized by cost, with a deadline guard.
    """
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return safe finite values to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus for high uncertainty
    # This helps escape local optima
    exploration = 0.1 * sigma
    
    # Combined score: EI + exploration, normalized by cost
    score = (ei + exploration) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
