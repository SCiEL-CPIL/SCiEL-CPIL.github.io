"""Rewritten acquisition for cost-aware BO on 2D Rosenbrock (5 instances)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Probability of Improvement per unit cost,
    with a small exploration bonus for high-uncertainty regions.
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std predictions in objective units
        f_best: lowest observed value
        cost: (2000,) evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    # Deadline check: if we're running out of time, return a simple score
    if time.time() - _START_TIME > DEADLINE_SECONDS:
        return np.zeros_like(mu)
    
    # Avoid division by zero / negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Small exploration bonus: reward high uncertainty
    # This helps escape local minima in the Rosenbrock valley
    exploration_bonus = 0.05 * sigma_safe
    
    # Combine: PI + exploration bonus, scaled by 1/cost
    # Cost ranges from 1 to 2, so 1/cost ranges from 0.5 to 1.0
    score = (pi + exploration_bonus) / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
