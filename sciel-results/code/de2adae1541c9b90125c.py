"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


# Track start time for deadline guard
_ACQ_START_TIME = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with a deadline guard.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP mean predictions
    sigma : array-like, shape (n,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more desirable)
    """
    # Deadline guard: if we're close to the 300s limit, return a safe fallback
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > 295:  # safety margin of 5s
        # Fallback: simple EI per cost
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
    
    # Compute Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component: explore where uncertainty is high
    # Use a moderate beta value
    beta = 0.5
    ucb = -mu + beta * sigma  # Note: we're minimizing, so lower mu is better
    
    # Cost-aware combination:
    # 1. EI per unit cost (primary driver)
    # 2. UCB per unit cost (exploration)
    # 3. Small bonus for cheaper points to encourage cost efficiency
    
    cost_safe = cost + 1e-9
    
    # Normalize cost to [0, 1] range for the bonus term
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.ones_like(cost)
    
    # Combination: weighted sum of EI/cost and UCB/cost
    # EI/cost is the primary signal, UCB/cost adds exploration
    ei_cost = ei / cost_safe
    ucb_cost = ucb / cost_safe
    
    # Weight EI more heavily, with a small exploration component
    w_ei = 1.0
    w_ucb = 0.3
    
    # Small bonus for cheaper points (encourages cost efficiency)
    cost_bonus = -0.1 * cost_norm
    
    scores = w_ei * ei_cost + w_ucb * ucb_cost + cost_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
