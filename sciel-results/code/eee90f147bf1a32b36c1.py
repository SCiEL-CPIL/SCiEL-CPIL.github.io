import numpy as np
import time
from scipy.stats import norm

# Global state for timing
_acq_state = {
    'start_time': None
}

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 290:
        # Fallback: simple LCB
        eps = 1e-9
        s = np.maximum(sigma, eps)
        c = np.maximum(cost, eps)
        score = -(mu + 2.5 * s) / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Gaussian Process UCB with budget-scaled exploration
    # mu, sigma, cost are arrays of shape (2000,)
    # f_best is a scalar
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Compute total cost to determine budget progress
    total_cost = np.sum(cost)
    
    # Budget is 30, so progress goes from 0 to 1 as cost approaches 30
    # Clamp to [0, 1]
    budget_progress = np.clip(total_cost / 30.0, 0.0, 1.0)
    
    # Exploration coefficient decays linearly from 3.0 to 0.5
    # High exploration early, low exploration late
    kappa = 3.0 - 2.5 * budget_progress
    
    # UCB for minimization: we want to minimize, so we look for low values
    # UCB = mu - kappa * sigma (lower bound of confidence interval)
    # We want to maximize this score (higher = more promising)
    ucb = mu - kappa * s
    
    # Cost-aware scaling: divide by convex combination of cost and constant
    # This balances exploitation of low-cost regions against exploration
    # Use a convex combination: alpha * cost + (1-alpha) * 1.0
    # alpha = 0.5 gives equal weight to cost and constant
    alpha = 0.5
    cost_factor = alpha * c + (1 - alpha) * 1.0
    
    # Score = UCB / cost_factor
    # For minimization, lower mu is better, so we negate to get higher scores for better candidates
    score = -ucb / cost_factor
    
    # Clean up
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
