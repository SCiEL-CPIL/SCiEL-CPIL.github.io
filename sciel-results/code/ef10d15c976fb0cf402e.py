"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def _greedy_fill_fast(capacity, weights, values, excluded_idx):
    """Fast greedy fill with multiple policies, returns best total value."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    mask = np.ones(n, dtype=bool)
    mask[excluded_idx] = False
    mask &= weights <= capacity
    if not np.any(mask):
        return 0.0

    idxs = np.where(mask)[0]
    w = weights[idxs]
    v = values[idxs]

    best_total = 0.0

    # Policy 0: density
    order = np.argsort(-(v / w))
    total = 0.0
    cap = capacity
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 0:
                break
    if total > best_total:
        best_total = total

    # Policy 1: value
    order = np.argsort(-v)
    total = 0.0
    cap = capacity
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 0:
                break
    if total > best_total:
        best_total = total

    # Policy 2: blend (0.7 density + 0.3 normalized value)
    max_val = np.max(v)
    if max_val > 0:
        norm_v = v / max_val
    else:
        norm_v = np.zeros_like(v)
    blend_score = 0.7 * (v / w) + 0.3 * norm_v
    order = np.argsort(-blend_score)
    total = 0.0
    cap = capacity
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 0:
                break
    if total > best_total:
        best_total = total

    return best_total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Select next item using lookahead with multiple greedy policies."""
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check time budget
    if time.monotonic() - _START > _TOTAL_BUDGET:
        return -1
    
    # Only consider items that fit
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    # Compute value density
    density = feasible_values / feasible_weights
    
    # Blend: 70% density + 30% normalized value
    max_val = np.max(feasible_values)
    if max_val > 0:
        norm_val = feasible_values / max_val
    else:
        norm_val = np.zeros_like(feasible_values)
    
    score = 0.7 * density + 0.3 * norm_val
    
    # Lookahead: for each candidate, estimate total value using multiple greedy policies
    # Use the improved _greedy_fill_fast which tries density, value, and blend
    if n <= 80 and time.monotonic() - _START < _TOTAL_BUDGET * 0.85:
        best_score = -1
        best_idx = -1
        for i in feasible_indices:
            w_i = feasible_weights[i]
            v_i = feasible_values[i]
            rem_cap = remaining_capacity - w_i
            
            # Use the multi-policy greedy fill
            fill_val = _greedy_fill_fast(rem_cap, remaining_weights, remaining_values, i)
            total_val = v_i + fill_val
            
            # Combine with density score for tie-breaking
            combined = total_val + 0.1 * score[i]
            if combined > best_score:
                best_score = combined
                best_idx = i
        
        if best_idx >= 0:
            return int(best_idx)
    
    # Fallback: best blended score
    best_local = int(np.argmax(score))
    return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END
