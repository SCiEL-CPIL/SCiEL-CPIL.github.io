"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80  # safety margin before 90s limit
    
    p = 1009
    n = 1008  # vertices are 1..1008
    
    # Precompute modular inverses: inv[x] = x^(-1) mod p
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse elements (x * x % p == 1) cannot appear in any valid interval
    # These are x = 1 and x = p-1 = 1008
    self_inverse = set()
    for x in range(1, p):
        if inv[x] == x:
            self_inverse.add(x)
    
    # Two-pointer sliding window to find longest valid interval
    # An interval [L, R] is valid if for all x, y in [L, R], x*y % p != 1
    # This means no pair (x, inv[x]) can both be in the interval (including x == inv[x])
    
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        # If R is a self-inverse, it cannot be in any valid interval
        if R in self_inverse:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in the current window [L, R-1], we have a conflict
        # Move L past invR
        if L <= invR < R:
            L = invR + 1
        
        # Also, if invR >= R, check if it could cause issues later
        # But for now, the window [L, R] is valid by construction
        
        # Check current window length
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
    
    # Verify the result (sanity check)
    start = best_start
    length = best_len
    end = start + length - 1
    
    # Verify no forbidden pairs
    valid = True
    for x in range(start, end + 1):
        if x in self_inverse:
            valid = False
            break
        ix = inv[x]
        if start <= ix <= end:
            valid = False
            break
    
    if not valid:
        # Fallback: return a known valid small interval
        # [2, 2] is valid since 2*2=4 != 1 mod 1009
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
