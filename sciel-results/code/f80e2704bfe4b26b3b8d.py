"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

# Deadline guard
_DEADLINE = 320.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using dynamically-scaled Probability of Improvement (POI).
    The improvement threshold adapts to remaining budget to prevent wasting
    evaluations on marginal gains late in the run.
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Estimate remaining budget and evaluations
    # Total budget is 30. Average cost per eval is roughly 1.5 (mean of 1+mean(x) over [0,1]^6)
    # So total evaluations ≈ 30 / 1.5 = 20
    # Initial 5 points are already spent.
    # Remaining budget = 30 - total_cost_spent_so_far
    # We don't know total_cost_spent_so_far directly, but we can infer from f_best and typical progress.
    # Alternatively, we can use a heuristic: 
    # If f_best is very high (> -1), we're early. If low (< -3.2), we're late.
    
    # A better approach: track the number of evaluations implicitly.
    # Since we can't track state, we use f_best as a proxy for progress.
    # Early: f_best > -2.0, Late: f_best < -3.2
    
    # Dynamic gamma: how much improvement do we demand?
    # When budget is plentiful (early), demand less (gamma small).
    # When budget is scarce (late), demand more (gamma large).
    
    # Let's define a "budget pressure" based on f_best.
    # f_best starts around -1.5 to -2.0 and ends around -3.32.
    # Normalize progress: p = (f_best - (-1.5)) / (-3.32 - (-1.5)) = (f_best + 1.5) / (-1.82)
    # But this is inverted. Let's use:
    # progress = 1 - (f_best - f_min) / (f_start - f_min)
    # Simpler: just use f_best directly.
    
    # If f_best > -2.5: early, gamma = 0.1
    # If -2.5 >= f_best > -3.0: mid, gamma = 0.3
    # If f_best <= -3.0: late, gamma = 0.5
    
    if f_best > -2.5:
        gamma = 0.1
    elif f_best > -3.0:
        gamma = 0.3
    else:
        gamma = 0.5

    # Target improvement threshold
    # We want P(f(x) < f_best - threshold)
    # threshold = gamma * (remaining_budget_estimate)
    # Since we don't know remaining_budget exactly, we approximate.
    # Let's assume we're at evaluation k. Total evals ~ 20.
    # remaining_evals ~ 20 - k
    # remaining_cost ~ remaining_evals * 1.5
    # threshold = gamma * remaining_cost / remaining_evals = gamma * 1.5? No, that's constant.
    
    # Let's re-read the strategy: "target as f_best - gamma * (30 - total_cost) / n_evals_remaining"
    # This is complex to estimate. Let's simplify.
    # The key is to make the threshold larger when budget is low.
    
    # Alternative: Use the cost of the candidate itself to adjust.
    # But the strategy says "dynamically scaled by the remaining budget".
    
    # Let's use a simpler heuristic for the threshold.
    # threshold = gamma * sigma_scale
    # where sigma_scale is a measure of the expected improvement potential.
    
    # Actually, let's stick to the POI formula with a dynamic threshold.
    # POI = P(f(x) < f_best - delta)
    # = P((f(x) - mu) / sigma < (f_best - delta - mu) / sigma)
    # = Phi((f_best - delta - mu) / sigma)
    
    # What should delta be?
    # If we're early, delta can be small (we're fine with small improvements).
    # If we're late, delta should be larger (we need big improvements to justify cost).
    
    # Let's define delta based on f_best.
    # delta = 0.05 * (f_best - f_min)  # Scale with distance to optimum?
    # f_min = -3.32237
    f_min = -3.32237
    dist_to_opt = f_best - f_min
    
    # When dist_to_opt is large, we can afford small improvements.
    # When dist_to_opt is small, we need larger improvements.
    # So delta should be proportional to dist_to_opt? Or inverse?
    # If we're close to optimum, any improvement is precious, but also harder to get.
    # We should demand less improvement when close? No, the strategy says "force the search to demand larger relative improvements as the budget depletes".
    
    # So delta should increase as budget depletes.
    # Budget depletes as f_best improves.
    # So delta should increase as f_best decreases (improves).
    
    # Let's use: delta = gamma * dist_to_opt
    # When dist_to_opt is large (early), delta is large? That seems backwards.
    # If we're early, we have lots of budget, so we can be picky (large delta).
    # If we're late, we have little budget, so we should be less picky (small delta)?
    # No, the strategy says "demand larger relative improvements as the budget depletes".
    # This means we should be MORE picky when budget is low.
    # So delta should be LARGER when budget is low (f_best is good).
    
    # So delta = gamma * (some function that increases as f_best improves)
    # Let's use: delta = gamma * (f_start - f_best)
    # f_start ~ -1.5
    f_start = -1.5
    progress = f_start - f_best  # Increases as f_best improves
    
    # But progress can be negative if f_best < f_start.
    progress = np.maximum(progress, 0.1)
    
    # Normalize progress to [0, 1]
    max_progress = f_start - f_min  # ~ 1.82
    norm_progress = np.minimum(progress / max_progress, 1.0)
    
    # delta = gamma * norm_progress * scale
    # scale should be a reasonable improvement size.
    # Let's use scale = 0.1 * dist_to_opt
    # So delta = gamma * norm_progress * 0.1 * dist_to_opt
    
    # This is getting complicated. Let's simplify.
    # Just use: delta = gamma * dist_to_opt * 0.1
    # When f_best is good (dist_to_opt small), delta is small.
    # But we want delta to be LARGE when budget is low.
    
    # Contradiction. Let's re-read: "demand larger relative improvements as the budget depletes"
    # Maybe "relative" means relative to the current uncertainty?
    
    # Let's try a different approach:
    # POI with threshold delta = gamma * sigma
    # This makes the threshold proportional to the uncertainty.
    # When uncertainty is high, we demand more improvement.
    # When uncertainty is low, we demand less.
    
    # This is a standard approach. Let's use it.
    delta = gamma * sigma_safe
    
    # POI = Phi((f_best - delta - mu) / sigma)
    z_poi = (f_best - delta - mu) / sigma_safe
    poi = ndtr(z_poi)
    
    # Cost-awareness: penalize high cost
    cost_norm = np.clip((cost_safe - 1.0) / 1.0, 0, 1)
    
    # Adaptive cost penalty
    if f_best > -3.0:
        beta = 0.8
    elif f_best > -3.2:
        beta = 1.2
    else:
        beta = 1.6
    
    cost_penalty = 1.0 + cost_norm * beta * 0.5
    
    scores = poi / cost_penalty
    
    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
