import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_call_count = 0
_total_cost = 0.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using dynamic UCB with adaptive exploration budget.
    """
    global _call_count, _total_cost
    _call_count += 1

    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - 5.0:
        # Fallback: Greedy exploitation (minimize mu / cost)
        scores = -mu / np.maximum(cost, 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Estimate remaining budget: start with 5 points, each costs ~1.5 on average
    # Total budget is 30. We track approximate remaining budget.
    # Since we don't know exactly which points were picked, we estimate based on call count.
    # Initial 5 points + _call_count evaluations.
    # Average cost per eval is roughly 1.5 (midpoint of [0,1] scaling).
    # Total cost so far approx 5*1.5 + _call_count*1.5 = 7.5 + 1.5*_call_count
    # But f_best is given, so we are in the loop.
    # Let's use a simpler proxy: remaining budget = 30 - estimated_cost.
    # estimated_cost = 7.5 + 1.5 * _call_count
    # If _call_count is 0, we just started the loop after initial 5 points.
    
    est_cost = 7.5 + 1.5 * _call_count
    remaining_budget = max(0.1, 30.0 - est_cost)
    
    # Dynamic beta for UCB:
    # Early on (high remaining budget), we explore more (higher beta).
    # Late on (low remaining budget), we exploit more (lower beta).
    # Scale beta by remaining_budget.
    # A typical UCB beta is 2.0. Let's make it adaptive.
    # If remaining_budget is ~22.5 (start), beta ~ 2.0 * (22.5/22.5) = 2.0
    # If remaining_budget is ~1.0 (end), beta ~ 2.0 * (1.0/22.5) ~ 0.09
    beta = 2.0 * (remaining_budget / 22.5)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB: mu + beta * sigma
    # We want to minimize (mu + beta * sigma) / cost
    # So we maximize - (mu + beta * sigma) / cost
    
    ucb_values = mu + beta * sigma_safe
    scores = -ucb_values / np.maximum(cost, 1e-9)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
