"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    
    # Sweep line approach
    # L is the minimum valid left boundary for an interval ending at R
    L = 1
    
    for R in range(1, n + 1):
        y = inv[R]
        
        # If y is in [L, R], it's a conflict.
        # Specifically, if L <= y <= R, then y is in the interval.
        # If y == R, it's a self-inverse.
        # If y < R, it's a pair.
        # In both cases, we must push L past y.
        if L <= y <= R:
            L = y + 1
            
        # Now [L, R] should be valid.
        if L <= R:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
