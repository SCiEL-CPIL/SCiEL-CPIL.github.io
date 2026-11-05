import time
import random
import numpy as np
import math

def _precompute_ap_data(n):
    """Precompute indices and masks for all APs to speed up evaluation."""
    data = []
    for d in range(1, n):
        max_len_d = (n - 1) // d + 1
        # Create indices for all starts r in 0..d-1
        # Rows: r (0 to d-1)
        # Cols: k (0 to max_len_d-1)
        
        # Using broadcasting
        r_idx = np.arange(d)[:, None]
        k_idx = np.arange(max_len_d)[None, :]
        
        flat_indices = r_idx + k_idx * d
        valid_mask = flat_indices < n
        
        data.append((flat_indices, valid_mask))
    return data

def _eval_all(signs, ap_data):
    """
    Computes the maximum absolute signed sum on ANY arithmetic progression.
    Returns: (max_disc, worst_d, worst_r)
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int32)
    global_max = 1
    worst_d = 1
    worst_r = 0
    
    for d in range(1, n):
        flat_indices, valid_mask = ap_data[d-1]
        
        # Gather signs
        # Use safe indices to avoid out of bounds
        safe_indices = np.where(valid_mask, flat_indices, 0)
        S_vals = signs_arr[safe_indices]
        S_vals[~valid_mask] = 0
        
        # Cumulative sum
        C = np.cumsum(S_vals, axis=1)
        
        # Max subarray sum
        MinPref = np.minimum.accumulate(C, axis=1)
        MaxSub = C - MinPref
        row_maxs = np.max(MaxSub, axis=1)
        
        # Min subarray sum
        MaxPref = np.maximum.accumulate(C, axis=1)
        MinSub = C - MaxPref
        row_mins = np.min(MinSub, axis=1)
        
        current_max_d = np.max(row_maxs)
        current_min_d = np.min(row_mins)
        
        disc_d = max(int(current_max_d), int(-current_min_d))
        
        if disc_d > global_max:
            global_max = disc_d
            if current_max_d >= -current_min_d:
                worst_r = int(np.argmax(row_maxs))
            else:
                worst_r = int(np.argmin(row_mins))
            worst_d = d
            
        # Optimization: if d > n//2 and global_max >= 2, we can break?
        # No, because a later d might have a larger discrepancy?
        # For d > n//2, max length is 2. Max disc is 2.
        # So if global_max >= 2, no subsequent d (which has max disc <= 2) can exceed it.
        # So we can break.
        if d > n // 2 and global_max >= 2:
            break
            
    return global_max, worst_d, worst_r

def _get_worst_ap_indices(n, d, r):
    """Get the list of indices for the AP starting at r with step d."""
    return list(range(r, n, d))

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    n = 256
    rng = random.Random(42)
    
    # Precompute AP data
    ap_data = _precompute_ap_data(n)
    
    # Thue-Morse initialization
    signs = [0] * n
    for i in range(n):
        cnt = bin(i).count('1')
        signs[i] = 1 if cnt % 2 == 0 else -1
        
    current_disc, worst_d, worst_r = _eval_all(signs, ap_data)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Simulated Annealing / Hill Climbing
    temp = 5.0
    cool_rate = 0.999
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        
        # Decide move
        if rng.random() < 0.7 and worst_d > 0:
            # Targeted move: flip a bit in the worst AP
            ap_indices = _get_worst_ap_indices(n, worst_d, worst_r)
            if len(ap_indices) > 0:
                flip_idx = ap_indices[rng.randint(0, len(ap_indices) - 1)]
            else:
                flip_idx = rng.randint(0, n - 1)
        else:
            # Random move
            flip_idx = rng.randint(0, n - 1)
            
        old_val = signs[flip_idx]
        new_val = -old_val
        signs[flip_idx] = new_val
        
        # Evaluate
        new_disc, new_worst_d, new_worst_r = _eval_all(signs, ap_data)
        
        delta = new_disc - current_disc
        
        # Accept if better or with probability exp(-delta/temp)
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 0.1)):
            current_disc = new_disc
            worst_d = new_worst_d
            worst_r = new_worst_r
            
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
                no_improve = 0
            else:
                no_improve += 1
                
            temp *= cool_rate
        else:
            # Revert
            signs[flip_idx] = old_val
            no_improve += 1
            temp *= cool_rate
            
        if best_disc == 1:
            break
            
        if no_improve > max_no_improve:
            # Reset temp
            temp = 5.0
            no_improve = 0
            
    return best_signs
