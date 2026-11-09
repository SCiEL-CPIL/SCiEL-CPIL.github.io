# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = time.time() + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement
    with a modified cost penalty (square root) to favor the inner domain
    where the true minimum is located (inner 80%).
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Use square root of cost in the denominator.
    # Rationale: The true minimum is in the inner 80% of the domain, which has
    # higher evaluation costs (1.1 to 1.9) than the boundaries (1.0 to 1.1).
    # Dividing by cost (linear) over-penalizes the inner region.
    # Dividing by sqrt(cost) reduces this bias, allowing the search to
    # consider high-cost, high-potential regions more fairly.
    scores = ei / np.sqrt(cost)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
