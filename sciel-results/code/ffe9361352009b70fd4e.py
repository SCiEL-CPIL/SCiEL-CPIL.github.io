"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 320.0:
        # Return PI/cost as a safe default
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        z = (f_best - mu_safe) / np.maximum(sigma_safe, 1e-8)
        pi = norm.cdf(z)
        return pi / np.maximum(cost_safe, 1e-8)
    
    # Ensure proper array types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is at least a tiny positive value
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    n = len(mu)
    
    # Cost-Adjusted LCB with dynamic exploration based on remaining budget
    # The key insight: as we approach the cost limit, we should exploit more
    
    # Estimate remaining budget fraction
    # Total budget is 30, initial 5 points cost ~5*1.5=7.5 on average
    # We don't know exact total cost used, but we can estimate from call_count
    # Each call evaluates 1 point, so roughly call_count evaluations done
    # Average cost per eval is about 1.5, so total cost ≈ 5 + call_count * 1.5
    # But this is rough. Let's use a simpler heuristic based on call_count.
    # After 5 initial points, each acquisition call adds 1 eval.
    # So total evals ≈ 5 + (call_count - 1) = 4 + call_count
    # Total cost ≈ sum of costs, roughly (4 + call_count) * 1.5
    # Remaining budget ≈ 30 - (4 + call_count) * 1.5
    
    total_cost_estimate = 5.0 + max(0, _acq_state['call_count'] - 1) * 1.5
    remaining_budget = max(30.0 - total_cost_estimate, 0.5)
    
    # Fraction of budget remaining (clamped to [0.05, 1.0])
    budget_frac = np.clip(remaining_budget / 30.0, 0.05, 1.0)
    
    # Dynamic kappa: scale exploration by sqrt of remaining budget fraction
    # When budget is plentiful (budget_frac ~ 1), kappa is larger (more exploration)
    # When budget is scarce (budget_frac ~ 0.05), kappa is smaller (more exploitation)
    base_kappa = 1.0
    kappa = base_kappa * np.sqrt(budget_frac)
    
    # Additional adaptive component: if f_best is close to GP's best prediction,
    # reduce exploration further
    best_mu = np.min(mu)
    gap = f_best - best_mu
    mu_std = np.std(mu)
    if mu_std > 1e-8:
        # Normalize gap by std
        norm_gap = gap / mu_std
        # If gap is small relative to uncertainty, we're close to optimum
        if norm_gap < 1.0:
            # Reduce kappa further
            kappa *= 0.5
        elif norm_gap > 3.0:
            # Large gap, allow more exploration
            kappa *= 1.2
    
    # Compute LCB for minimization
    lcb = mu - kappa * sigma
    
    # Cost adjustment: divide by cost to account for evaluation expense
    # Use a smooth cost penalty: score = -lcb / cost^alpha
    # alpha = 1.0 for linear cost scaling
    scores = -lcb / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
