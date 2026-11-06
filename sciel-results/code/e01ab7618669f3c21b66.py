"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    # Iterate over all possible start points
    for a in range(1, n + 1):
        # Self-inverse elements 1 and 1008 cannot be in the interval
        if a == 1 or a == 1008:
            continue
        
        b = a
        # Extend the interval as far as possible
        while b < n:
            y = b + 1
            # Check if y is self-inverse
            if y == 1 or y == 1008:
                break
            # Check if the inverse of y is in the current interval [a, b]
            # If inv[y] is in [a, b], then there exists x in [a, b] such that x*y % p == 1
            if a <= inv[y] <= b:
                break
            b = y
        
        current_len = b - a + 1
        if current_len > best_len:
            best_len = current_len
            best_start = a
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
