"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    best_len = 0
    best_start = 1
    
    # Iterate over all possible start positions
    for s in range(1, n + 1):
        # Iterate over possible end positions
        # We can break early if the interval becomes invalid
        for e in range(s, n + 1):
            # Check if e and inv[e] are both in [s, e]
            # If inv[e] is in [s, e], then the pair (e, inv[e]) is in the interval
            # which violates the condition e * inv[e] % 1009 == 1
            if s <= inv[e] <= e:
                break
            
            # If we reach here, [s, e] is a valid interval
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
