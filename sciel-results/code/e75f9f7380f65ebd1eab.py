"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    # inv[x] is the number y in [1, 1008] such that x*y % 1009 == 1
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # Self-inverses: x such that x*x % 1009 == 1
    # These are 1 and 1008 (since 1008 = -1 mod 1009, (-1)^2 = 1)
    self_inv = {1, 1008}
    
    best_len = 0
    best_start = 1
    
    # Sliding window approach
    # L is the left pointer of the current valid window
    # We iterate R (right pointer) from 1 to 1008
    L = 1
    
    for R in range(1, MOD):
        # If R is a self-inverse, it cannot be in any valid interval
        # So we must move L past R
        if R in self_inv:
            L = R + 1
            continue
        
        # Otherwise, check if inv(R) is in the current window [L, R-1]
        # If it is, we have a conflict between R and inv(R)
        # We must move L to be strictly greater than inv(R)
        inv_R = inv[R]
        if L <= inv_R < R:
            L = inv_R + 1
            
        # Now [L, R] is a valid interval
        # Update best
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
