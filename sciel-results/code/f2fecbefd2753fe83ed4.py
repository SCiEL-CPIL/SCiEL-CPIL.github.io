import time
import numpy as np

def construct():
    MOD = 1009
    N = 1008
    start_time = time.time()
    deadline = 80  # safety margin
    
    # Precompute inverses
    # x from 1 to 1008
    x_vals = np.arange(1, MOD)
    inv = np.array([pow(x, MOD - 2, MOD) for x in x_vals])
    
    best_start = 2
    best_len = 0
    
    # We iterate L from 2 to 1007
    # For each L, we find max R
    # R must be <= 1007
    # Condition: for all x in [L, R], inv[x] not in [L, R]
    # Equivalent to: R < min(inv[x] for x in [L, 1007] if inv[x] >= L)
    
    for L in range(2, 1008):
        if time.time() - start_time > deadline:
            break
        
        # x indices in inv array: 0 to 1007 correspond to 1 to 1008
        # We are interested in x in [L, 1007]
        # These correspond to indices [L-1, 1006] in inv array
        # Wait, inv[0] is inv(1), inv[1] is inv(2), ..., inv[1007] is inv(1008)
        # We want x in [L, 1007].
        # So indices are L-1 to 1006.
        
        # Slice inv for x in [L, 1007]
        # Note: inv[1007] is inv(1008). We should exclude x=1008?
        # The loop in original code went to 1007 (exclusive 1008).
        # So x in [L, 1007].
        # Indices: L-1 to 1006.
        
        if L > 1007:
            continue
            
        sub_inv = inv[L-1:1007] # indices L-1 ... 1006
        
        if len(sub_inv) == 0:
            continue
            
        # Find min inv[x] such that inv[x] >= L
        mask = sub_inv >= L
        if np.any(mask):
            K = np.min(sub_inv[mask])
            R_max = K - 1
        else:
            R_max = 1007
        
        if R_max >= L:
            length = R_max - L + 1
            if length > best_len:
                best_len = length
                best_start = L
    
    return [best_start, best_len]
