"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5 seconds margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: scalar, best (lowest) observed function value
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - scores: array of shape (2000,) with acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe fallback: random or uniform scores
        return np.random.rand(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Safety: handle zero or negative cost
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost squared to prioritize cheaper evaluations more strongly
    # This helps maximize the number of evaluations within the fixed budget
    scores = ei / (cost_safe ** 2)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
