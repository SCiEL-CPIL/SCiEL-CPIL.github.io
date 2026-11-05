"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses and self-inverse status
    inv = [0] * (p)
    is_self_inv = [False] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        if (x * x) % p == 1:
            is_self_inv[x] = True
    
    best_len = 0
    best_start = 1
    
    # Iterate over all possible starts
    for s in range(1, n + 1):
        if is_self_inv[s]:
            continue
        
        e = s
        # Extend the interval as far as possible
        # We break when adding e_candidate creates a conflict
        for e_candidate in range(s + 1, n + 1):
            if is_self_inv[e_candidate]:
                break
            
            inv_ec = inv[e_candidate]
            # Conflict if inv[e_candidate] is in [s, e_candidate - 1]
            if s <= inv_ec < e_candidate:
                break
            
            e = e_candidate
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
