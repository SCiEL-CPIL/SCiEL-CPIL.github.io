"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    self_inv = [False] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        if x * x % p == 1:
            self_inv[x] = True
    
    best_start = 2
    best_len = 1
    
    # Local variables for speed
    inv_local = inv
    self_inv_local = self_inv
    best_len_local = best_len
    best_start_local = best_start
    
    for s in range(1, 1009):
        if time.time() > deadline:
            break
        
        if 1008 - s + 1 <= best_len_local:
            continue
            
        if self_inv_local[s]:
            continue
        
        e = s
        # Extend e as far as possible
        while e < 1008:
            new_e = e + 1
            if self_inv_local[new_e]:
                break
            # Check if inv[new_e] is in [s, new_e]
            # If it is, we break. If not, we extend.
            if not (s <= inv_local[new_e] <= new_e):
                e = new_e
            else:
                break
        
        length = e - s + 1
        if length > best_len_local:
            best_len_local = length
            best_start_local = s
    
    return [best_start_local, best_len_local]
# EVOLVE-BLOCK-END
