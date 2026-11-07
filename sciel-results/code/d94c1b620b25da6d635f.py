import numpy as np
import time

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    
    Uses a Cost-Effective Upper Confidence Bound (CE-UCB) strategy.
    Score = (-mu + beta * sigma) / cost
    where beta is scaled by the square root of the remaining budget.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0
    
    _call_count += 1
    
    # Deadline guard: if we are running out of time, return a safe exploitation score
    # to ensure we don't crash or timeout in the middle of the loop.
    if time.time() > _deadline:
        # Fallback: pure exploitation (negative mean) divided by cost
        # This is a valid finite score.
        eps = 1e-10
        return -mu / (cost + eps)
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # Estimate remaining budget
    # Initial cost: 5 points * avg cost 1.5 = 7.5
    # Each iteration adds roughly 1.5 cost on average (candidates are uniform, best is picked)
    # This is an estimate, but sufficient for dynamic beta scaling.
    initial_cost = 7.5
    avg_step_cost = 1.5
    estimated_spent = initial_cost + _call_count * avg_step_cost
    total_budget = 30.0
    remaining_budget = max(total_budget - estimated_spent, 0.1)
    
    # Dynamic beta scaling
    # Base beta of 1.0, scaled by sqrt(remaining_budget)
    # This intensifies exploration when budget is high (early) and reduces it as budget depletes.
    beta = 1.0 * np.sqrt(remaining_budget)
    
    # Compute UCB score for minimization: -mu + beta * sigma
    # We want higher scores to be better, so we return this value.
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware normalization
    # Divide by cost to prioritize cheaper evaluations
    scores = ucb_score / (cost + eps)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
