"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Cost-Adjusted Expected Improvement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_EPS = 1e-12
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) acquisition function.
    
    Strategy:
    1. Use Expected Improvement (EI) as the core acquisition metric, which is 
       theoretically optimal for minimizing regret given a Gaussian process model.
    2. Adjust for cost by dividing the EI by the evaluation cost of each candidate.
       This effectively maximizes the expected improvement per unit cost.
    3. This approach is stateless, robust, and adapts naturally to the budget 
       constraints without requiring manual tuning of exploration coefficients 
       or global state tracking.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    t0 = time.time()
    
    # Ensure sigma is non-negative and cost is positive
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Phi is the standard normal CDF, phi is the standard normal PDF
    
    # Avoid division by zero or very small sigma
    safe_sigma = np.where(sigma > _EPS, sigma, 1.0)
    
    z = (f_best - mu) / safe_sigma
    
    # For very large negative z (mu >> f_best), EI is approx 0
    # For very large positive z (mu << f_best), EI is approx (f_best - mu)
    # scipy.stats.norm handles this well, but we can clip z for numerical stability if needed
    # z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted EI
    # We want to maximize EI / cost
    # This encourages finding improvements in cheap regions
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return score
        
    return score

# EVOLVE-BLOCK-END
