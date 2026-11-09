import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0

def _elapsed_ok():
    return time.time() - _start_time < _DEADLINE

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
    
    # Precompute orders for various α
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    orders = {}
    for alpha in alphas:
        if alpha == 0.0:
            ratio = values[feas_mask].copy()
        else:
            fw = weights[feas_mask]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = values[feas_mask] / np.power(fw, alpha)
        ratio[weights[feas_mask] == 0] = np.inf
        order = np.argsort(-ratio)
        orders[alpha] = order
    
    # Fast estimate: use α=0.5 for all candidates
    order_fast = orders[0.5]
    fast_scores = np.zeros(nf)
    for idx in range(nf):
        i = feas_indices[idx]
        cap_after = remaining_capacity - weights[i]
        fill_val = 0.0
        sim_cap = cap_after
        for j in order_fast:
            orig_j = feas_indices[j]
            if orig_j == i:
                continue
            if weights[orig_j] <= sim_cap:
                sim_cap -= weights[orig_j]
                fill_val += values[orig_j]
        fast_scores[idx] = values[i] + fill_val
    
    # Top K candidates
    K = min(30, nf)
    top_k = np.argsort(-fast_scores)[:K]
    
    # Thorough evaluation for top K
    best_total = -1.0
    best_idx = -1
    for idx in top_k:
        if not _elapsed_ok():
            break
        i = feas_indices[idx]
        cap_after = remaining_capacity - weights[i]
        base_val = values[i]
        best_fill = 0.0
        for alpha in alphas:
            order = orders[alpha]
            fill_val = 0.0
            sim_cap = cap_after
            for j in order:
                orig_j = feas_indices[j]
                if orig_j == i:
                    continue
                if weights[orig_j] <= sim_cap:
                    sim_cap -= weights[orig_j]
                    fill_val += values[orig_j]
            if fill_val > best_fill:
                best_fill = fill_val
        total = base_val + best_fill
        if total > best_total:
            best_total = total
            best_idx = i
    
    return int(best_idx)
