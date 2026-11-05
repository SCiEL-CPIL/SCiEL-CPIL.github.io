"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    p = 1009
    n = 1008  # We consider elements 1..1008
    
    # Precompute modular inverses for all x in 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverses: x such that x*x % p == 1
    # These are x = 1 and x = p-1 = 1008
    self_inv = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inv.add(x)
    
    # Sliding window to find longest valid interval
    best_len = 0
    best_start = 1
    L = 1  # Left endpoint of current window
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # If R is a self-inverse, it cannot be in any valid interval
        # We must move L past R
        if R in self_inv:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in the current window [L, R-1], we have a conflict
        # Move L past invR to resolve it
        if L <= invR < R:
            L = invR + 1
        
        # Now [L, R] should be valid (by construction)
        # But we need to verify: could there be another conflict?
        # No, because all elements in [L, R-1] were mutually compatible,
        # and the only element that conflicts with R is inv(R).
        # If inv(R) < L, no conflict. If inv(R) >= R, no conflict in window.
        # If inv(R) was in [L, R-1], we moved L past it.
        
        cur_len = R - L + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = L
    
    if best_len == 0:
        # Fallback: find any valid single element
        for x in range(1, n + 1):
            if x not in self_inv:
                best_start = x
                best_len = 1
                break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
