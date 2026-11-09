"""Rewritten program with cost-aware LCB acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: kill after 310s wall-clock to stay safely under 330s limit
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0

# Track total evaluations to adapt the exploration coefficient
_eval_count = 0
_TOTAL_BUDGET = 30.0  # cost budget mentioned in the task


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound acquisition for Bayesian optimization
    on the Levy 2D function.
    
    Uses an adaptive exploration coefficient that starts high (for exploration)
    and decays as the cost budget is consumed (shifting to exploitation).
    
    Args:
        mu: (2000,) array of GP means in objective units
        sigma: (2000,) array of GP standard deviations in objective units
        f_best: float, best observed value so far
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of acquisition values (higher = more promising)
    """
    global _eval_count

    # Deadline guard: if we're getting close to the limit, return a simple fallback
    if time.time() > _DEADLINE:
        return np.zeros_like(mu, dtype=float)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Estimate how much of the budget has been used
    # Each call represents one acquisition evaluation cycle
    _eval_count += 1
    
    # Estimate consumed budget: start with 5 initial points, each adding 1 eval
    # Total cost budget is ~30, starting cost is ~5 (from 5 random points)
    # So we've used roughly (eval_count / (30-5)) of the remaining budget
    # But more precisely, track by expected cost per evaluation
    # Since we don't have exact running total, estimate: 
    # Each iteration adds ~1 unit of cost on average (mean cost ~1 + 0.5 = 1.5)
    # So consumed cost ≈ 5 + 1.5 * _eval_count, capped at 30
    est_consumed = min(5.0 + 1.5 * _eval_count, _TOTAL_BUDGET)
    frac_remaining = max(1.0 - est_consumed / _TOTAL_BUDGET, 0.0)
    
    # Adaptive exploration coefficient:
    # Start with xi=3.0 for strong exploration, decay to xi=0.5 for exploitation
    # The coefficient scales with remaining budget to shift focus
    xi = 0.5 + 2.5 * (1.0 - frac_remaining)
    
    # LCB acquisition: higher when mu is low (good) and sigma is high (uncertain)
    # score = -(mu + xi * sigma) / cost
    # This favors low predicted values and high uncertainty, penalized by cost
    lcb_raw = mu + xi * sigma_safe
    scores = -lcb_raw / cost_safe
    
    # Add a small pure uncertainty term to keep exploration alive even late
    # Weight it so it's negligible when we're in exploitation mode
    uncertainty_weight = 0.1 * frac_remaining
    scores += uncertainty_weight * sigma_safe / cost_safe
    
    # Normalize scores to prevent extreme values from dominating
    # Use a robust scaling: shift by median and scale by IQR
    finite_mask = np.isfinite(scores)
    if np.count_nonzero(finite_mask) > 10:
        med = np.median(scores[finite_mask])
        q1, q3 = np.percentile(scores[finite_mask], [25, 75])
        iqr = q3 - q1
        if iqr > 1e-12:
            scores = (scores - med) / iqr
        else:
            scores = scores - med
    
    # Ensure all scores are finite
    scores = np.where(finite_mask, scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
