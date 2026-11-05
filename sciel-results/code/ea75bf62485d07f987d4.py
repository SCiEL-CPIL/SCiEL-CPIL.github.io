"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin within 60s limit
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_len = 1
    
    # Iterate over possible starts
    for L in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # If L is self-inverse, skip
        if inv[L] == L:
            continue
        
        # Extend R from L upward
        max_R = L
        
        for R in range(L, N + 1):
            if time.time() > deadline:
                break
                
            # Check if adding R creates a violation
            # Violation if inv[R] is in [L, R]
            if L <= inv[R] <= R:
                break
            
            max_R = R
        
        length = max_R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
