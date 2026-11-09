"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_fill_value(w, v, capacity):
    """Compute total value from greedy density fill. Returns (value, count_selected)."""
    if len(w) == 0 or capacity <= 0:
        return 0.0, 0
    # Filter feasible items
    mask = w <= capacity
    if not np.any(mask):
        return 0.0, 0
    w = w[mask]
    v = v[mask]
    if len(w) == 0:
        return 0.0, 0
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    # Sort by density descending
    order = np.argsort(-densities)
    total_val = 0.0
    remaining_cap = capacity
    count = 0
    for idx in order:
        if remaining_cap <= 0:
            break
        wi = w[idx]
        vi = v[idx]
        if wi <= remaining_cap:
            total_val += vi
            remaining_cap -= wi
            count += 1
    return total_val, count


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    if c <= 0:
        return -1
    
    # Filter to feasible items only
    feasible_mask = w <= c
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_indices]
    v_f = v[feasible_indices]
    n_f = len(w_f)
    
    if n_f == 0:
        return -1
    
    # Compute densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v_f / w_f
    # Handle zero-weight items: if weight is 0 and value > 0, density is inf
    # If weight is 0 and value == 0, density is nan; treat as 0
    densities = np.where(np.isinf(densities), np.finfo(float).max, densities)
    densities = np.where(np.isnan(densities), 0.0, densities)
    
    # Sort by density descending
    order = np.argsort(-densities)
    
    # 1-step lookahead over top candidates
    # Consider top K candidates for lookahead
    K = min(30, n_f)
    top_candidates = order[:K]
    
    best_score = -1.0
    best_idx = -1
    
    for cand_pos in range(len(top_candidates)):
        cand = top_candidates[cand_pos]
        wi = w_f[cand]
        vi = v_f[cand]
        
        if wi > c:
            continue
        
        # After picking this item, greedily fill the rest
        remaining_cap = c - wi
        if remaining_cap <= 0:
            total_val = vi
        else:
            # Build arrays excluding the candidate
            mask_remove = np.ones(n_f, dtype=bool)
            mask_remove[cand] = False
            w_rest = w_f[mask_remove]
            v_rest = v_f[mask_remove]
            
            if len(w_rest) == 0:
                total_val = vi
            else:
                # Greedy fill on remaining
                feasible_rest = w_rest <= remaining_cap
                if not np.any(feasible_rest):
                    total_val = vi
                else:
                    w_r = w_rest[feasible_rest]
                    v_r = v_rest[feasible_rest]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        d_r = v_r / w_r
                    d_r = np.where(np.isinf(d_r), np.finfo(float).max, d_r)
                    d_r = np.where(np.isnan(d_r), 0.0, d_r)
                    order_r = np.argsort(-d_r)
                    total_val = vi
                    rem_cap = remaining_cap
                    for idx_r in order_r:
                        if rem_cap <= 0:
                            break
                        w_ir = w_r[idx_r]
                        v_ir = v_r[idx_r]
                        if w_ir <= rem_cap:
                            total_val += v_ir
                            rem_cap -= w_ir
        
        # Score: total value from this choice
        if total_val > best_score:
            best_score = total_val
            best_idx = feasible_indices[cand]
    
    if best_idx == -1:
        # Fallback: pick highest density feasible item
        best_idx = feasible_indices[top_candidates[0]]
    
    return int(best_idx)
# EVOLVE-BLOCK-END
