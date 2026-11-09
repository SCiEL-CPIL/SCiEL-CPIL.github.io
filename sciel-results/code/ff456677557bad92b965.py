import numpy as np
import time

# Global deadline guard
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Initialize timer on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've used more than 140 seconds
    if _get_elapsed() > 140:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    C = float(remaining_capacity)
    
    # Filter to feasible items only
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    fw = w[feasible_mask]
    fv = v[feasible_mask]
    fn = len(fw)
    
    if fn == 0:
        return -1
    
    # If only one feasible item, take it
    if fn == 1:
        return int(feasible_indices[0])
    
    # One-step look-ahead: for each candidate, estimate total value
    # if we take it and then greedily fill the rest by density
    # Precompute density order for the full set
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(fw > 0, fv / fw, np.inf)
    
    # Sort by density descending
    density_order = np.argsort(-densities)
    
    best_score = -1
    best_local_idx = 0
    
    for local_i in range(fn):
        # Check time
        if _get_elapsed() > 140:
            return int(feasible_indices[best_local_idx])
        
        wi = fw[local_i]
        vi = fv[local_i]
        remaining_C = C - wi
        
        # Greedily fill remaining capacity with density order, skipping item i
        total_val = vi
        rem_cap = remaining_C
        
        for idx in density_order:
            if idx == local_i:
                continue
            if fw[idx] <= rem_cap:
                total_val += fv[idx]
                rem_cap -= fw[idx]
            if rem_cap <= 0:
                break
        
        if total_val > best_score:
            best_score = total_val
            best_local_idx = local_i
    
    return int(feasible_indices[best_local_idx])
