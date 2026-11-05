"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # Sliding window [l, r]
    # We maintain that [l, r] is a valid interval.
    # We want to maximize r - l + 1.
    
    l = 1
    max_len = 0
    best_start = 1
    
    for r in range(1, N + 1):
        # If r is self-inverse, it cannot be in any valid interval.
        # So we must exclude it.
        if r == 1 or r == 1008:
            l = r + 1
            continue
            
        inv_r = inv[r]
        
        # If inv_r is in the current window [l, r-1], we have a conflict.
        # We must move l to be > inv_r.
        # Note: inv_r cannot be r because r is not self-inverse here.
        # If inv_r < l, it's not in the window.
        # If inv_r > r, it's not in the window (since window ends at r).
        # So conflict only if l <= inv_r <= r-1.
        
        if l <= inv_r <= r - 1:
            l = inv_r + 1
            
        # Now [l, r] is valid.
        current_len = r - l + 1
        if current_len > max_len:
            max_len = current_len
            best_start = l
            
    return [best_start, max_len]
# EVOLVE-BLOCK-END
