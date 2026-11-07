# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition with budget-risk adjustment.
    # Base: Expected Improvement per unit cost.
    # Adjustment: down-weight candidates that risk exhausting the budget.
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Per-unit-cost EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Budget-risk adjustment:
    # Total budget is 30. We penalize candidates whose cumulative cost
    # (current total + this candidate's cost) approaches the limit.
    # total_cost is not directly available, but cost is the marginal cost.
    # We approximate risk by the candidate's own cost relative to remaining budget.
    # Since we don't know total_cost, we use a heuristic:
    # Higher cost candidates are more risky. We scale by exp(-lambda * cost / budget_scale)
    # where budget_scale is a fraction of total budget (30).
    
    # Use a risk-aversion parameter. Higher lambda = more risk-averse.
    lambda_risk = 2.0
    budget_scale = 10.0  # Scale for cost penalty
    
    risk_factor = np.exp(-lambda_risk * cost_safe / budget_scale)
    
    # Combine
    scores = ei_per_cost * risk_factor
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
