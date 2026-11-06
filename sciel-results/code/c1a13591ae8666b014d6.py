"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    best_start = 1
    best_len = 0
    
    left = 1
    
    for right in range(1, N + 1):
        # Check if 'right' is a self-inverse (1 or 1008)
        if inv[right] == right:
            # Cannot include this element in any valid interval
            # Reset window to start after this element
            left = right + 1
            continue
        
        inv_r = inv[right]
        
        # If the inverse of 'right' is in the current window [left, right-1],
        # we must move 'left' to be strictly greater than inv_r
        if inv_r >= left:
            left = inv_r + 1
        
        # Now [left, right] is a valid interval
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left
    
    # Fallback (should not happen as N>=1 and valid intervals exist)
    if best_len == 0:
        return [2, 1]
        
    return [best_start, best_len]
# EVOLVE-BLOCK-END
