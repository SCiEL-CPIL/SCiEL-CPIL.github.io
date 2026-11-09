"""Knapsack heuristic with UCB-inspired scoring and 1-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 2.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def _greedy_fill(remaining_capacity, weights, values, exclude_mask):
    """
    Greedy fill using density order, excluding specified items.
    Returns total value achievable.
    """
    if remaining_capacity <= 0:
        return 0.0
    
    n = len(weights)
    # Compute density and sort
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)
    order = np.argsort(-densities)
    
    sim_cap = remaining_capacity
    sim_val = 0.0
    
    for item_idx in order:
        if exclude_mask[item_idx]:
            continue
        w = weights[item_idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[item_idx]
        if sim_cap <= 0:
            break
    
    return sim_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Safety check: if we've been running too long, fall back to simple greedy
    if not _elapsed_ok():
        if len(remaining_weights) == 0 or remaining_capacity <= 0:
            return -1
        weights = np.asarray(remaining_weights, dtype=float)
        values = np.asarray(remaining_values, dtype=float)
        feasible = np.where(weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        dens = values[feasible] / np.maximum(weights[feasible], 1e-10)
        best_local = np.argmax(dens)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 1:
        return int(feasible_indices[0])

    # Compute base scores for UCB-inspired selection
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)
    
    # Normalize density to [0,1] range for scoring
    max_dens = np.max(densities[feasible_mask]) if np.any(feasible_mask) else 1.0
    if max_dens <= 0:
        max_dens = 1.0
    
    # Compute weight synergy score: items that leave good capacity for subsequent items
    # Higher score for items whose weight is a "nice fraction" of remaining capacity
    # This helps identify items that might unlock better combinations
    cap_ratio = weights / remaining_capacity
    # Synergy score peaks around 0.3-0.5 of capacity (empirically good for knapsack)
    synergy_score = 1.0 - np.abs(cap_ratio - 0.4)
    synergy_score = np.clip(synergy_score, 0, 1)
    
    # UCB-style score: balance exploitation (density) and exploration (synergy)
    # alpha controls exploration vs exploitation tradeoff
    alpha = 0.3
    ucb_scores = (1 - alpha) * (densities / max_dens) + alpha * synergy_score
    
    # Rank feasible items by UCB score
    feasible_ucb = ucb_scores[feasible_mask]
    ranked_feasible = feasible_indices[np.argsort(-feasible_ucb)]
    
    # Limit candidates to top-K for efficiency
    K = min(15, n_feasible)
    candidates = ranked_feasible[:K]
    
    # For each candidate, estimate total value with 1-step lookahead
    best_total_val = -1.0
    best_idx = -1
    
    for idx in candidates:
        if not _elapsed_ok():
            break
            
        w_i = weights[idx]
        v_i = values[idx]
        remaining_cap = remaining_capacity - w_i
        
        if remaining_cap <= 0:
            total_val = v_i
        else:
            # Simulate greedy fill excluding this item
            exclude_mask = np.zeros(n, dtype=bool)
            exclude_mask[idx] = True
            fill_val = _greedy_fill(remaining_cap, weights, values, exclude_mask)
            total_val = v_i + fill_val
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = idx
            
    if best_idx == -1:
        # Fallback: pick the first candidate
        best_idx = int(candidates[0])
        
    return int(best_idx)
# EVOLVE-BLOCK-END
