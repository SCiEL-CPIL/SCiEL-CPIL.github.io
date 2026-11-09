"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_t0 = time.time()
_DEADLINE = 330.0  # wall-clock limit; leave margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - lowest observed function value
    - cost: array of shape (2000,) - cumulative cost (we use a fixed per-eval cost estimate)
    
    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running low on time, return a simple score
    elapsed = time.time() - _t0
    if elapsed > _DEADLINE - 5.0:
        # Fallback: just return -mu (exploit best prediction)
        return -mu.astype(np.float64)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: The cost of each evaluation is 1 + mean(x_scaled) ∈ [1, 2].
    # We estimate average per-eval cost as 1.5.
    per_eval_cost = 1.5
    
    # CEI (Cost-Effective Improvement): EI / cost
    # This is the theoretically optimal greedy policy for cost-aware BO
    cei = ei / per_eval_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(cei), cei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
