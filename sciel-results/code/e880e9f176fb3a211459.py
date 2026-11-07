"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track wall-clock deadline (330s limit, use 305s margin)
_T0 = time.time()
_DEADLINE = 305.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def _fast_ei(mu, sigma, f_best):
    """Fast expected improvement for minimization."""
    sigma = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _fast_ucb(mu, sigma, beta):
    """Upper confidence bound for minimization (lower is better)."""
    sigma = np.maximum(sigma, 1e-10)
    return -mu + beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of Styblinski-Tang 2D.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if not _time_ok():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-6)
        ei = _fast_ei(mu, sigma_safe, float(f_best))
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite and positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)

    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0

    # Infer progress: f_best starts near 0 (origin) and goes toward -78.3323
    # progress = 0 at start, 1 at optimum
    # Use a more aggressive mapping since we have limited budget
    # ST-2d: f(0,0)=0, min=-78.3323
    # In practice, we'll see values like -20, -40, -60, -70, -75, etc.
    # Map so that progress reaches ~1.0 when f_best < -50
    progress = np.clip(-f_best_val / 50.0, 0.0, 1.0)
    
    # Also consider how many evaluations we've likely done based on cost
    # Total cost starts at ~5 (5 initial points * avg cost ~1.5) and goes to 30
    # So we've used roughly (total_cost - 5) / 25 of our budget
    # But we don't have total_cost directly, just per-point cost
    # We can estimate from the number of points with cost > 0
    # Actually, the cost array represents the cost of each of the 2000 candidate points
    # So we can't directly infer evaluation count from this
    # Let's stick with f_best-based progress

    # Adaptive parameters:
    # Early (progress ~ 0): high exploration (high beta, high UCB weight)
    # Late (progress ~ 1): high exploitation (low beta, high EI weight)
    
    # Beta for UCB: start high, decrease
    # With limited budget, we want to explore a bit more than typical
    beta = 1.5 - 1.2 * progress  # 1.5 early, 0.3 late
    
    # Weight for UCB vs EI
    # Early: more UCB, Late: more EI
    w_ucb = 0.5 - 0.35 * progress  # 0.5 early, 0.15 late
    w_ei = 1.0 - w_ucb             # 0.5 early, 0.85 late

    # Compute scores
    ei = _fast_ei(mu, sigma_safe, f_best_val)
    ucb = _fast_ucb(mu, sigma_safe, beta)

    # Cost-aware: divide by cost to prefer cheaper points
    # But don't over-penalize; use a softer cost adjustment
    # cost ranges from 1.0 (center) to 2.0 (corners)
    # Use cost^0.5 for a milder penalty
    cost_adj = np.power(cost_safe, 0.5)
    
    score_ei = ei / cost_adj
    score_ucb = ucb / cost_adj

    score = w_ei * score_ei + w_ucb * score_ucb

    # Additional: strongly penalize high-cost points in late game
    # When we're close to optimum, we want to exploit nearby cheap points
    if progress > 0.5:
        # Penalize points with cost > 1.3 more heavily
        cost_excess = np.maximum(cost - 1.3, 0.0)
        penalty = 0.3 * (progress - 0.5) * cost_excess ** 2
        score = score - penalty

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Normalize to a reasonable range to avoid numerical issues
    # This helps the evaluator pick the best point reliably
    score_max = np.max(score)
    score_min = np.min(score)
    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    else:
        score = np.zeros_like(score)

    return score
# EVOLVE-BLOCK-END
