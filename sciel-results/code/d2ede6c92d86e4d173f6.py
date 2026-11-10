"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 290.0
_t0 = time.time()


def _greedy_solve_fast(prizes, weights, capacity, exclude_mask=None):
    """Fast greedy: pick best prize/total_weight item that fits, repeat."""
    n = prizes.shape[1]
    item_prizes = prizes.sum(axis=0)
    
    if exclude_mask is not None:
        eligible = ~exclude_mask.copy()
    else:
        eligible = np.ones(n, dtype=bool)
    
    remaining = capacity * np.ones(5)
    total_prize = 0.0
    selected = np.zeros(n, dtype=bool)
    
    for _ in range(n):
        if not eligible.any():
            break
        
        # Vectorized: find best ratio among eligible items that fit
        # An item fits if all its weights <= remaining
        fits = np.all(weights <= remaining[:, None], axis=0)  # shape (100,)
        candidates = eligible & fits
        
        if not candidates.any():
            break
        
        ratios = np.where(candidates, item_prizes / (weights.sum(axis=0) + 1e-9), -1.0)
        best_idx = int(np.argmax(ratios))
        
        if ratios[best_idx] < 0:
            break
        
        selected[best_idx] = True
        total_prize += item_prizes[best_idx]
        remaining -= weights[:, best_idx]
        eligible[best_idx] = False
    
    return total_prize, selected


def heuristics(prizes, weights):
    t_start = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Wrong shape")
    
    capacity = 25.0
    n = 100
    
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    item_prizes = prizes.sum(axis=0)
    total_w = weights.sum(axis=0)
    
    # Baseline greedy
    base_prize, base_selected = _greedy_solve_fast(prizes, weights, capacity)
    
    # Regret for items in baseline only
    regret = np.zeros(n, dtype=np.float64)
    baseline_items = np.where(base_selected)[0]
    
    for j in baseline_items:
        if time.time() - t_start > _DEADLINE - 15:
            break
        exclude = np.zeros(n, dtype=bool)
        exclude[j] = True
        excl_prize, _ = _greedy_solve_fast(prizes, weights, capacity, exclude_mask=exclude)
        regret[j] = max(0.0, base_prize - excl_prize)
    
    # Additional signals
    density = item_prizes / (total_w + 1e-9)
    
    slack_per_constraint = capacity - weights
    min_slack = slack_per_constraint.min(axis=0)
    max_slack = min_slack.max() + 1e-9
    slack_norm = np.maximum(min_slack, 0) / max_slack
    
    ratios = prizes / weights
    min_ratio = ratios.min(axis=0)
    
    # Normalize
    max_regret = regret.max() + 1e-9
    regret_norm = regret / max_regret
    
    density_scaled = np.power(density, 1.5)
    max_dens = density_scaled.max() + 1e-9
    density_scaled = density_scaled / max_dens
    
    min_ratio_scaled = min_ratio / (min_ratio.max() + 1e-9)
    
    heuristic = (
        0.4 * regret_norm +
        0.25 * density_scaled +
        0.2 * slack_norm +
        0.15 * min_ratio_scaled
    )
    
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
