import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 140.0

def _calc_frac_bound(global_order, excluded, w, v, C):
    """
    Calculate fractional knapsack value for items in global_order,
    excluding items in 'excluded' set, with capacity C.
    """
    total_val = 0.0
    rem_cap = C
    for idx in global_order:
        if idx in excluded:
            continue
        wi = w[idx]
        vi = v[idx]
        if wi <= rem_cap:
            total_val += vi
            rem_cap -= wi
            if rem_cap <= 0:
                break
        else:
            if wi > 0:
                total_val += (rem_cap / wi) * vi
            break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        if len(remaining_weights) == 0:
            return -1
        # Fallback: Simple greedy
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = v[feasible] / w[feasible]
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights)
    v = np.array(remaining_values)
    C = remaining_capacity
    
    # Pre-compute densities and global order
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities[w == 0] = np.inf
    global_order = np.argsort(-densities, kind='stable')
    
    # Identify feasible items
    feasible_mask = w <= C
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    # Phase 1: 1-Step Lookahead
    # Score each feasible item i: v[i] + frac_bound(remaining, C - w[i])
    scores_1 = {}
    for i in feasible_indices:
        C_rem = C - w[i]
        if C_rem < 0:
            continue
        excluded = {int(i)}
        bound = _calc_frac_bound(global_order, excluded, w, v, C_rem)
        scores_1[int(i)] = v[i] + bound
        
    if not scores_1:
        return -1

    # Get top K candidates from Phase 1
    # Sort by score descending
    sorted_items = sorted(scores_1.items(), key=lambda x: x[1], reverse=True)
    K = 5 # Number of candidates to explore in 2-step
    top_candidates = [idx for idx, _ in sorted_items[:K]]
    
    best_score = -np.inf
    best_idx = -1
    
    # Phase 2: 2-Step Lookahead for top candidates
    for i in top_candidates:
        C1 = C - w[i]
        # Find feasible second items
        # We can just iterate through all feasible items except i
        # To limit computation, we can just check all, N is small
        excluded_i = {int(i)}
        
        # We want to try a few 'j's. Let's try all feasible j, or top few by density.
        # Given N=100, trying all j is 100 * 100 (inner loop) = 10k ops.
        # So trying all j is fine.
        
        best_score_for_i = -np.inf
        for j in feasible_indices:
            if int(j) == int(i):
                continue
            if w[j] > C1:
                continue
            
            C2 = C1 - w[j]
            if C2 < 0:
                continue
                
            excluded_ij = {int(i), int(j)}
            bound = _calc_frac_bound(global_order, excluded_ij, w, v, C2)
            score = v[i] + v[j] + bound
            
            if score > best_score_for_i:
                best_score_for_i = score
                
        if best_score_for_i > best_score:
            best_score = best_score_for_i
            best_idx = int(i)
            
    # If 2-step didn't find anything (shouldn't happen if 1-step did), fallback to best 1-step
    if best_idx == -1:
        best_idx = top_candidates[0]
        
    return best_idx
