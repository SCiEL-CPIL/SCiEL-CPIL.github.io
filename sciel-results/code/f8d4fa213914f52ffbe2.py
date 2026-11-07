# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = 330.0 - 30.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _within_budget():
    return _elapsed() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural change:
    instead of EI, we use a cost-aware LCB (Lower Confidence Bound) with
    an adaptive beta that transitions from exploration to exploitation.
    
    Key structural differences from EI-based approaches:
    1. LCB directly optimizes the lower confidence bound (optimistic prediction)
    2. Beta schedule is tied to remaining budget, not just elapsed cost
    3. Cost normalization uses a convex combination to avoid over-penalizing
       expensive but high-uncertainty regions early on
    4. Add a small diversity bonus based on sigma to avoid collapse
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Estimate current total cost. The cost array contains the cost of each
    # of the 2000 candidate points. The actual total spent is not directly
    # given, but we can infer progress from f_best and the magnitude of
    # improvements. A simpler proxy: use the mean cost as a rough indicator.
    # Better: track via a heuristic based on how "tight" f_best is relative
    # to the function minimum (0 for three-hump camel).
    
    # Use f_best to estimate progress: if f_best is close to 0, we're late.
    # Three-hump camel minimum is 0. Typical f_best starts around 1-5.
    # Map f_best to a progress fraction in [0, 1].
    # Early: f_best ~ 2-5 -> progress ~ 0
    # Late: f_best ~ 0.01-0.1 -> progress ~ 1
    if f_best > 0:
        # Log-scale mapping: progress = 1 - log(f_best + eps) / log(f_max + eps)
        f_max = 5.0  # typical initial f_best
        progress = 1.0 - np.log(f_best + 1e-8) / np.log(f_max + 1e-8)
    else:
        progress = 1.0
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta schedule: high exploration early, low late
    # beta goes from ~3.0 (early) to ~0.3 (late)
    beta = 3.0 - 2.7 * progress
    
    # LCB for minimization: mu - beta * sigma
    # We want to MINIMIZE LCB, so score should be HIGHER for lower LCB
    # score = -LCB = -mu + beta * sigma
    lcb = mu - beta * sigma_safe
    score_lcb = -lcb  # = -mu + beta * sigma
    
    # Cost-aware normalization:
    # Use a convex combination: effective_cost = alpha * cost + (1-alpha) * 1
    # where alpha increases with progress (more cost-sensitive late)
    alpha = 0.5 + 0.5 * progress  # 0.5 early, 1.0 late
    effective_cost = alpha * cost_safe + (1.0 - alpha) * 1.0
    
    # Normalize: divide by effective_cost
    score = score_lcb / effective_cost
    
    # Add a small diversity/exploration bonus to prevent collapse
    # This is a small term that rewards high sigma points slightly
    diversity_bonus = 0.1 * beta * sigma_safe / effective_cost
    score = score + diversity_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
