"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    p = 1009
    n = 1008  # interval is in [1, 1008]
    
    # Precompute modular inverses for all x in [1, p-1]
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Two-pointer sweep to find the longest valid interval.
    # A window [left, right] is valid iff for all x in [left, right], inv[x] is outside [left, right].
    # When extending right, the only new conflict is between right and inv[right].
    # If inv[right] is in [left, right], we must push left to inv[right] + 1.
    
    left = 1
    best_start = 1
    best_length = 0
    
    for right in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # If right is a self-inverse (right^2 ≡ 1 mod p), it can't be in any valid interval
        if inv[right] == right:
            left = right + 1
            continue
        
        # If inv[right] falls within [left, right], we must exclude it by advancing left
        if inv[right] >= left:
            left = inv[right] + 1
        
        # Now [left, right] is valid
        length = right - left + 1
        if length > best_length:
            best_length = length
            best_start = left
    
    # Ensure we return a non-empty interval
    if best_length <= 0:
        # Fallback: find any single valid element
        for x in range(1, n + 1):
            if inv[x] != x:
                best_start = x
                best_length = 1
                break
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
