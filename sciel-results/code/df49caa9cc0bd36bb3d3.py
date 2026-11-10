import numpy as np
import time

_start_time = None

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety check: if we've used 115 seconds, fallback to simple density
    if time.time() - _start_time > 115:
        if len(remaining_weights) == 0:
            return -1
        w = np.asarray(remaining_weights, dtype=float)
        v = np.asarray(remaining_values, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities[np.where(w == 0)] = 0
        return int(np.argmax(densities))

    n = len(remaining_weights)
    if n == 0:
        return -1
    
    if n == 1:
        return 0

    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities[np.where(w == 0)] = 0
    
    # Sort by density desc, then weight asc
    # lexsort sorts by the last key first.
    # So keys are (w, -densities).
    # Primary: -densities (desc), Secondary: w (asc)
    order = np.lexsort((w, -densities))
    
    best_score = -1.0
    best_idx = -1
    
    # Pre-compute w and v as lists for faster access in inner loop?
    # Numpy scalar access is slower than list access in Python loops.
    w_list = w.tolist()
    v_list = v.tolist()
    order_list = order.tolist()
    
    for i in range(n):
        wi = w_list[i]
        # Input is feasible, so wi <= c
        cap_left = c - wi
        total_val = v_list[i]
        
        for j in order_list:
            if j == i:
                continue
            wj = w_list[j]
            if wj <= cap_left:
                total_val += v_list[j]
                cap_left -= wj
            if cap_left <= 0:
                break
                
        if total_val > best_score:
            best_score = total_val
            best_idx = i
            
    return best_idx
