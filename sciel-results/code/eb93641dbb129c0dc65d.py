"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    p = 1009
    
    # Self-inverse elements: x^2 ≡ 1 mod p => x = 1 or x = p-1
    self_inv = {1, p - 1}
    
    # Precompute pairs {u, v} with u < v and u*v ≡ 1 mod p
    pairs = []
    for i in range(1, p):
        if i in self_inv:
            continue
        inv_i = pow(i, p - 2, p)
        if i < inv_i:
            pairs.append((i, inv_i))
            
    # min_v_for_u[u] = v if (u, v) is a pair, else INF
    INF = p + 10
    min_v_for_u = [INF] * p
    for u, v in pairs:
        min_v_for_u[u] = v
        
    best_start = 1
    best_length = 0
    current_min = INF
    
    # Iterate s from p-1 down to 1
    # V_min(s) = min { v | u >= s, (u, v) in pairs }
    for s in range(p - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Update current_min with pairs starting at s
        if min_v_for_u[s] < current_min:
            current_min = min_v_for_u[s]
            
        # Check validity for start s
        if s in self_inv:
            continue
            
        # e_max is constrained by:
        # 1. Global bound 1007 (to exclude 1008)
        # 2. Pair constraints: e < v for any active pair (u, v) with u >= s
        #    The tightest constraint is e < min_v_for_u_active
        #    So e_max = min(1007, current_min - 1)
        
        e_max = 1007
        if current_min < INF:
            e_max = min(e_max, current_min - 1)
            
        # Interval [s, e_max] must be non-empty and valid
        if e_max < s:
            continue
            
        length = e_max - s + 1
        if length > best_length:
            best_length = length
            best_start = s
            
    return [best_start, best_length]
# EVOLVE-BLOCK-END
