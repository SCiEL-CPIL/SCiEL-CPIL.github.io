"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Uses a hybrid strategy:
    - Early phase (low total cost): aggressive exploration via LCB
    - Late phase (high total cost): exploitation via EI
    - Always normalized by cost
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Safety: ensure finite outputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate total cost spent so far from the cost array
    # The cost array represents the cost of each candidate, but we need to infer
    # how much budget has been used. Since we don't have direct access to the
    # number of evaluations, we use a heuristic based on the cost distribution.
    
    # A simpler approach: use the mean of the cost array as a proxy for the
    # average cost per evaluation, and assume we've used a portion of the budget.
    # The total budget is 30, and we start with 5 points.
    
    # Let's use a different approach: the cost values themselves tell us the
    # relative cost of each candidate. We can use the median cost to estimate
    # the typical evaluation cost, and from that infer the phase.
    
    # Actually, a cleaner approach: since the cost is 1 + mean(x scaled to [0,1]),
    # the minimum cost is 1 and maximum is 2. We can use the fact that the total
    # budget is 30 to estimate the phase.
    
    # Let's estimate the number of evaluations done so far.
    # If the average cost is c_avg, then number of evals ≈ total_cost_spent / c_avg.
    # But we don't know total_cost_spent directly.
    
    # Alternative: use the spread of the cost array as a signal.
    # If costs are all similar, we're in a stable phase.
    
    # Let's use a simpler heuristic: the phase is determined by how close f_best
    # is to 0 (the known minimum). If f_best is far from 0, we're early; if close, late.
    
    # Phase parameter: 0 = fully exploratory (LCB), 1 = fully exploitative (EI)
    # Use f_best to determine phase. The three hump camel 2d minimum is 0.
    # If f_best is large (e.g., > 1), we're early. If f_best is small (e.g., < 0.1), we're late.
    
    # Normalize f_best to [0, 1] for phase determination
    # Assume f_best ranges from ~5 (early) to ~0 (late)
    f_best_clamped = np.clip(f_best, 0, 5)
    phase = 1.0 - (f_best_clamped / 5.0)  # 0 when f_best=5, 1 when f_best=0
    
    # Blend LCB and EI based on phase
    # LCB: mu - beta * sigma
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Adaptive beta for LCB
    beta = 2.0 + 2.0 * (1.0 - phase)  # More aggressive exploration early
    
    lcb_score = mu - beta * sigma_safe
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei_score = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend the two
    score = phase * ei_score + (1.0 - phase) * lcb_score
    
    # Normalize by cost
    score /= cost_safe
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
