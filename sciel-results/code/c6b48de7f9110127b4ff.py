# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = time.time() + 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and Expected Improvement,
    with a cost penalty to prefer low-cost evaluations.
    
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
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB (for minimization: lower mu + exploration is better)
    # beta controls exploration vs exploitation
    # Use a time-adaptive beta that decreases as we get more information
    # Since we don't have explicit iteration count, use a fixed reasonable value
    beta = 1.5
    ucb = mu - beta * sigma  # For minimization, we want low mu and high sigma
    
    # Combine EI and UCB
    # Normalize both to have similar scale
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
    
    # Blend: primarily EI with some UCB component
    score = 0.6 * ei / ei_max + 0.4 * (ucb_max - ucb) / ucb_range
    
    # Cost penalty: prefer lower cost evaluations
    # Use a power penalty to strongly favor low-cost regions
    cost_penalty = 1.0 / (cost ** 1.5 + 1e-9)
    
    # Final score
    final_score = score * cost_penalty
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
