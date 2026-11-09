import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0

def _elapsed_ok():
    return time.time() - _start_time < _DEADLINE

def _vectorized_greedy_fill(capacity, weights, values, order_indices, skip_idx):
    """Compute greedy fill value using numpy operations."""
    if len(order_indices) == 0:
        return 0.0
    # Create mask to exclude skip_idx
    mask = np.ones(len(order_indices), dtype=bool)
    skip_pos = np.where(order_indices == skip_idx)[0]
    if len(skip_pos) > 0:
        mask[skip_pos[0]] = False
    
    # Simulate greedy fill
    sim_cap = capacity
    fill_val = 0.0
    # Process items in order
    for j in order_indices:
        if j == skip_idx:
            continue
        w_j = weights[j]
        v_j = values[j]
        if w_j <= sim_cap:
            sim_cap -= w_j
            fill_val += v_j
        if sim_cap <= 0:
            break
    return fill_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if not _elapsed_ok():
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        mask = w <= remaining_capacity
        if not np.any(mask):
            return -1
        ratios = v[mask] / w[mask]
        return int(np.where(mask)[0][np.argmax(ratios)])
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    feas_mask = weights <= remaining_capacity
    if not np.any(feas_mask):
        return -1
    
    feas_indices = np.where(feas_mask)[0]
    nf = len(feas_indices)
    
    # Precompute orders for various α - more diverse strategies
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    orders = {}
    for alpha in alphas:
        if alpha == 0.0:
            ratio = values[feas_mask].copy()
        else:
            fw = weights[feas_mask]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = values[feas_mask] / np.power(fw, alpha)
        # Handle zero weights
        zero_w = (weights[feas_mask] == 0)
        if np.any(zero_w):
            ratio[zero_w] = np.inf
        order = np.argsort(-ratio)
        orders[alpha] = order
    
    # Fast estimate: use α=0.5 for all candidates
    order_fast = orders[0.5]
    fast_scores = np.zeros(nf)
    for idx in range(nf):
        i = feas_indices[idx]
        cap_after = remaining_capacity - weights[i]
        if cap_after <= 0:
            fast_scores[idx] = values[i]
        else:
            fast_scores[idx] = values[i] + _vectorized_greedy_fill(cap_after, weights, values, feas_indices[order_fast], i)
    
    # Top K candidates - slightly larger K for better exploration
    K = min(50, nf)
    top_k = np.argsort(-fast_scores)[:K]
    
    # Thorough evaluation for top K using geometric mean of fill values (smooth combination)
    best_total = -1.0
    best_idx = -1
    for idx in top_k:
        if not _elapsed_ok():
            break
        i = feas_indices[idx]
        cap_after = remaining_capacity - weights[i]
        base_val = values[i]
        
        # Collect fill values across all orderings
        fill_vals = []
        for alpha in alphas:
            order = orders[alpha]
            fill_val = _vectorized_greedy_fill(cap_after, weights, values, feas_indices[order], i)
            fill_vals.append(fill_val)
        
        # Use geometric mean for smooth combination (avoids over-reliance on one ordering)
        fill_vals_arr = np.array(fill_vals)
        # Geometric mean: exp(mean(log(1+val))) - 1 to handle zeros
        log_vals = np.log1p(fill_vals_arr)
        geo_mean = np.exp(np.mean(log_vals)) - 1.0
        
        # Also track max for safety
        max_fill = np.max(fill_vals_arr)
        
        # Blend: 70% geometric mean + 30% max
        combined_fill = 0.7 * geo_mean + 0.3 * max_fill
        
        total = base_val + combined_fill
        if total > best_total:
            best_total = total
            best_idx = i
    
    return int(best_idx)
