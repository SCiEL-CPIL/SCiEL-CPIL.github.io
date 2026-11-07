"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Strategy: Cost-aware LCB with dynamic exploration coefficient and
    a cost penalty that scales with remaining budget.
    """
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Estimate progress based on f_best
    # Powell 4D on [-4,5]^4: initial values are around 10-20, minimum is 0
    # Use a normalized progress metric
    f_scale = max(abs(f_best), 1.0)
    progress = 1.0 - f_best / (f_scale + 1.0)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic beta: high exploration early, low exploitation late
    # beta controls the exploration-exploitation tradeoff in LCB
    beta = 0.3 + 1.2 * (1.0 - progress)
    
    # Cost-aware LCB: mu - beta * sigma - cost_penalty
    # The cost penalty should be stronger when budget is more constrained
    # Estimate remaining budget: total budget is 30, initial cost ~7.5
    # We don't know exactly how much has been spent, but we can use
    # the cost values themselves as a proxy
    
    # Mean cost of evaluated points gives a sense of typical evaluation cost
    mean_cost = np.mean(cost_safe)
    
    # Cost penalty: discourage expensive evaluations, especially later
    # Scale the penalty by progress so it becomes more important as budget depletes
    cost_weight = 0.1 + 0.4 * progress
    cost_penalty = cost_weight * np.log1p(cost_safe)
    
    # LCB score (higher is better for maximization)
    lcb = mu - beta * sigma_safe - cost_penalty
    
    # Ensure all outputs are finite
    lcb = np.where(np.isfinite(lcb), lcb, 0.0)
    
    # Check deadline
    if time.time() - t0 > deadline:
        return lcb
    
    return lcb
# EVOLVE-BLOCK-END
