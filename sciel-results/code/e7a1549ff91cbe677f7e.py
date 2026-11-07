# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 285.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization.
    
    Uses Lower Confidence Bound with cost normalization to directly target
    high-value (low f) low-cost regions. Stronger exploration via beta=2.0
    helps escape local minima in multi-modal Styblinski-Tang.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value (lower is better)
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = better candidate to evaluate next)
    """
    # Check deadline - if we're running out of time, return a simple score
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # Higher score = better, so we negate the LCB
    beta = 2.0
    lcb = mu - beta * sigma
    
    # Cost normalization: divide by cost to prefer low-cost evaluations
    # This naturally biases toward center-of-domain points near shifted minima
    cost_safe = cost + 1e-9
    score = -lcb / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
