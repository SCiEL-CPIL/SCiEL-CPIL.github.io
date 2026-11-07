import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization."""
    # Deadline guard: ensure we finish well before the 330s limit
    t_start = time.time()
    deadline = 330.0 - 15.0  # 15 second safety margin

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n == 0:
        return np.array([])

    # Safety: clamp sigma and cost to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Handle z overflow/underflow
    z = np.clip(z, -50, 50)

    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Stochastic shortest path formulation:
    # Score = EI / sqrt(remaining_budget + current_cost)
    # This dynamically relaxes cost penalty as budget depletes,
    # forcing exploitation in later stages.
    
    # Estimate remaining budget based on observed cost pattern
    # Total budget is 30, so remaining = 30 - sum of costs so far
    # But we don't have access to total cost, only per-candidate cost
    # We can estimate: if average cost per eval is ~1.5, and we've done ~10 evals,
    # spent ~15, remaining ~15. Use a heuristic: remaining = max(30 - 10*mean_cost, 1)
    mean_cost = np.mean(cost_safe)
    # Assume we're roughly in the middle of the budget
    # Use a dynamic estimate: remaining_budget = 30 - (cumulative cost estimate)
    # Since we don't have cumulative cost, use: remaining = 30 - n_evals * mean_cost
    # n_evals is unknown, but we can infer from cost distribution
    # Simpler: use remaining = 30 - 1.5 * n (assuming ~10 evals avg cost 1.5)
    # Better: use the cost values themselves as a proxy
    # The cost parameter here is the cost of evaluating each candidate
    # Total spent is not passed, so we use a heuristic based on cost magnitude
    
    # Heuristic: if mean cost is high, we're likely later in budget
    # remaining_budget = max(30 - 20 * mean_cost, 1.0)
    remaining_budget = max(30.0 - 20.0 * mean_cost, 1.0)
    
    # Denominator: sqrt(remaining_budget + current_cost)
    # This makes high-cost candidates less attractive when budget is plentiful,
    # but the penalty shrinks as remaining_budget decreases
    denominator = np.sqrt(remaining_budget + cost_safe)
    
    scores = ei / denominator
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add small exploration component for very uncertain regions
    # Only when EI is zero (no improvement expected)
    zero_ei_mask = ei < 1e-10
    if np.any(zero_ei_mask):
        uncertainty = sigma_safe[zero_ei_mask] / (np.max(sigma_safe[zero_ei_mask]) + 1e-9)
        scores[zero_ei_mask] = 0.05 * uncertainty / cost_safe[zero_ei_mask]

    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > deadline:
        return scores

    return scores
