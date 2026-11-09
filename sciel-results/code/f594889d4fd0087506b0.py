import numpy as np
from scipy.stats import norm

# Global state to track budget usage
_total_cost = 0.0
_call_count = 0
_initial_cost_estimate = 7.5 # 5 points * 1.5 avg cost
_budget_limit = 30.0

def acquisition(mu, sigma, f_best, cost):
    global _total_cost, _call_count
    
    # Estimate remaining budget
    # We don't know the exact cost of previous selections, but we can estimate.
    # Each call represents one iteration where 1 point is picked.
    # The cost of the picked point is roughly the mean of the costs or similar.
    # Let's assume average cost is 1.5.
    est_spent = _initial_cost_estimate + _call_count * 1.5
    remaining_budget = max(0.0, _budget_limit - est_spent)
    
    # Dynamic beta: scale exploration by sqrt(remaining budget)
    # Normalize remaining budget to [0, 1] roughly
    # Max remaining is ~22.5
    # Min remaining is 0
    # beta should be high when remaining is high.
    # Let's map remaining_budget to beta.
    # If remaining is 22.5, beta = 2.0 (aggressive)
    # If remaining is 0, beta = 0.5 (conservative)
    # Linear or sqrt mapping?
    # "scales ... by the square root of the remaining budget"
    # So term = sigma * sqrt(remaining_budget) ? No, usually beta * sigma.
    # Let beta = c * sqrt(remaining_budget)
    
    # Let's tune c.
    # If remaining = 22.5, sqrt = 4.7.
    # If we want beta ~ 2.0, c = 2.0 / 4.7 ~ 0.4.
    # If remaining = 1, sqrt = 1. beta = 0.4.
    # If remaining = 0, beta = 0.
    
    # To avoid zero exploration, add a small floor.
    # Also, ensure beta doesn't get too large initially if my estimate is off.
    
    c = 0.4
    beta = c * np.sqrt(remaining_budget) + 0.2 # Floor 0.2
    
    # LCB: mu - beta * sigma
    # We want to minimize, so lower is better.
    # Acquisition score should be higher for better points.
    # So we return -LCB? Or just use a positive metric?
    # "Return 2000 finite scores (higher = evaluate next)"
    # LCB is low when good (low mu, low sigma).
    # So we should return - (mu - beta * sigma) = beta * sigma - mu.
    # And normalize by cost.
    # Score = (beta * sigma - mu) / cost
    
    # Update counter
    _call_count += 1
    
    # Calculate scores
    # Handle sigma=0 case
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # LCB value (lower is better for the function value)
    lcb_values = mu - beta * sigma
    
    # Convert to "higher is better"
    scores = -lcb_values / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
