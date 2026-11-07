import numpy as np
import time
from scipy.special import erfc

_start_time = None
_total_cost = 0.0
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _total_cost, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 327.0:
        scores = f_best - mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Estimate remaining budget
    # Initial cost: 5 points, each cost in [1,2], so initial cost ~ 5 * 1.5 = 7.5
    # Total budget: 30
    # We track total cost consumed.
    # cost is the cost of evaluating the 2000 candidates.
    # The loop says: "While total cost is below 30, the evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    # So we need to track how much budget is left.
    # We don't know the exact total cost consumed so far, but we can estimate.
    # Each iteration consumes 1 evaluation cost (the best of 2000).
    # Let's assume we are in iteration k.
    # Actually, we can just use the number of calls as a proxy for iterations.
    # Initial 5 points are evaluated before the loop starts?
    # "Runs start with 5 uniform random points (cost counted)."
    # So call_count 1 corresponds to the first iteration after the initial 5 points.
    # Total cost so far = initial_cost + sum(costs of evaluated points).
    # We don't have the costs of evaluated points, only the costs of the current candidates.
    # But the costs are similar (uniform in [1,2]).
    # Let's estimate total cost consumed as: 5 * 1.5 + (call_count - 1) * 1.5 = 1.5 * (call_count + 4)
    # This is a rough estimate.
    # Remaining budget = 30 - estimated_total_cost.
    # If remaining budget is small, we should be more exploitative.
    
    estimated_total_cost = 1.5 * (_call_count + 4)
    remaining_budget = 30.0 - estimated_total_cost
    
    # If remaining budget is negative or very small, use pure exploitation
    if remaining_budget < 1.0:
        scores = f_best - mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Budget ratio: 1.0 when full, 0.0 when empty
    budget_ratio = np.clip(remaining_budget / 30.0, 0.0, 1.0)
    
    # Adaptive exploration weight (beta)
    # Early on (high budget_ratio), high exploration.
    # Late on (low budget_ratio), low exploration.
    # Also, if f_best is large, we need more exploration.
    # Let's combine budget ratio and f_best.
    
    # f_best scaling: Powell 4d can have large values.
    # Let's use a simple scaling.
    f_best_scale = f_best / (1.0 + f_best)
    
    # beta = (0.5 + 2.0 * f_best_scale) * (0.5 + 0.5 * budget_ratio)
    # This gives:
    # - Early (budget_ratio=1): beta = (0.5 + 2.0 * f_best_scale) * 1.0
    # - Late (budget_ratio=0): beta = (0.5 + 2.0 * f_best_scale) * 0.5
    # So even late, we have some exploration if f_best is large.
    
    beta = (0.5 + 2.0 * f_best_scale) * (0.5 + 0.5 * budget_ratio)
    
    # Cost-aware penalty
    # cost is in [1, 2].
    # We want to prefer cheaper evaluations when exploration is low (late game).
    # When exploration is high (early game), cost matters less.
    # lambda_cost = 0.1 * (1.0 - budget_ratio)
    # This means: early (budget_ratio=1) -> lambda=0, late (budget_ratio=0) -> lambda=0.1
    
    lambda_cost = 0.1 * (1.0 - budget_ratio)
    
    # Calculate scores
    # Maximize: f_best - mu + beta * sigma - lambda_cost * (cost - 1.5)
    ucb_term = f_best - mu + beta * sigma
    cost_penalty = lambda_cost * (cost - 1.5)
    
    scores = ucb_term - cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
