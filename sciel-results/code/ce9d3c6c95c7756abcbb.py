"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [start, start+length-1] contained in [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    We model this as finding the longest segment avoiding forbidden pairs (x, inv(x)) 
    and forbidden self-pairs (1, 1008).
    We use a two-pointer sliding window technique.
    """
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses using Fermat's Little Theorem
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    l = 1
    best_start = 1
    best_length = 0
    
    for r in range(1, N + 1):
        # Self-inverse elements (1 and 1008) cannot be in any valid interval
        if inv[r] == r:
            l = r + 1
            continue
            
        # If inv[r] is within the current window [l, r-1], we must advance l
        # to exclude it.
        if l <= inv[r] < r:
            l = inv[r] + 1
            
        # Update the best interval found so far
        current_length = r - l + 1
        if current_length > best_length:
            best_length = current_length
            best_start = l
            
    return [best_start, best_length]
# EVOLVE-BLOCK-END
