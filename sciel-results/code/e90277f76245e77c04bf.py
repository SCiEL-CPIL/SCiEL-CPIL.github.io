"""Rewritten acquisition with GP-UCB, budget-scaled exploration, and cost normalization."""
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

# Budget constants
_COST_BUDGET = 30.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: GP-UCB with budget-scaled exploration coefficient.
    
    Args:
        mu: array shape (2000,) - GP mean predictions in objective units
        sigma: array shape (2000,) - GP std dev predictions in objective units
        f_best: float - lowest observed value
        cost: array shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Time guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return np.ones_like(mu) / np.maximum(cost, 1e-9)
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Budget-scaled exploration coefficient
    # Estimate remaining budget from the cost array (cost of candidates already evaluated is not passed,
    # but we can infer progress from the magnitude of costs or use a heuristic)
    # Since we don't have explicit remaining budget, we use the mean cost as a proxy for progress:
    # Early on, costs are moderate; late on, we may have explored more.
    # Better approach: use the median cost as a baseline and scale beta by how "expensive" the current
    # candidates are relative to the budget.
    
    # Heuristic: estimate fraction of budget consumed based on the spread of costs.
    # If costs are concentrated near the low end, we may be early; if spread out, later.
    # Simpler: use a fixed decay schedule based on an internal counter would require state,
    # which we don't have. Instead, use the relationship between f_best and the prediction scale.
    
    # Alternative structural change: use the ratio of typical sigma to (f_best - min_mu) to gauge
    # how much uncertainty remains relative to improvement potential.
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = max(mu_max - mu_min, eps)
    
    # Scale beta by how much uncertainty there is relative to the function range
    # High uncertainty relative to range -> explore more (higher beta)
    # Low uncertainty relative to range -> exploit (lower beta)
    sigma_typical = np.median(s)
    uncertainty_ratio = sigma_typical / max(abs(f_best - mu_min), eps)
    
    # Clamp uncertainty ratio to [0, 2] to avoid extreme beta values
    uncertainty_ratio = np.clip(uncertainty_ratio, 0.0, 2.0)
    
    # Beta decays as uncertainty decreases: start at 3.0, decay to 0.5
    beta = 0.5 + 2.5 * uncertainty_ratio
    
    # GP-UCB for minimization: we want to minimize f(x)
    # UCB(x) = mu(x) + beta * sigma(x)  (upper bound on f)
    # We want to minimize the upper bound, so we maximize -UCB(x)
    # But we also want cost-awareness: prefer cheaper evaluations
    
    # Convex cost normalization:
    # Instead of pure division by cost, use a convex combination to balance
    # cost-efficiency with absolute improvement potential
    # cost_norm = alpha * cost + (1-alpha) * c_ref
    # where c_ref is a reference cost (median cost)
    c_ref = np.median(c)
    alpha = 0.5  # Balance between cost and reference
    
    cost_norm = alpha * c + (1.0 - alpha) * c_ref
    cost_norm = np.maximum(cost_norm, eps)
    
    # UCB score: higher is better for acquisition
    # For minimization, we want low mu and low UCB, so:
    # score = -(mu + beta * sigma) / cost_norm
    # But this penalizes high mu heavily. Better:
    # score = (f_best - mu - beta * sigma) / cost_norm
    # This measures expected improvement over f_best, discounted by cost
    
    # Expected improvement style: how much better than f_best could this be?
    # EI = max(0, f_best - mu - beta * sigma)
    # But we want smooth scores, so use a soft version:
    improvement = f_best - mu - beta * s
    
    # Normalize by cost
    ucb_score = improvement / cost_norm
    
    # Ensure finite output
    ucb_score = np.nan_to_num(ucb_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ucb_score
