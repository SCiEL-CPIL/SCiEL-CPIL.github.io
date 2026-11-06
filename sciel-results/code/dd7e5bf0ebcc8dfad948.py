"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses: inv[i] = i^{-1} mod 1009
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # Create list of conflict pairs (L_x, R_x)
    # L_x = min(x, inv[x]), R_x = max(x, inv[x])
    pairs = []
    for x in range(1, N + 1):
        ix = inv[x]
        L = x if x < ix else ix
        R = ix if x < ix else x
        pairs.append((R, L))
        
    # Sort by R (right endpoint of the conflict pair)
    # This allows us to efficiently determine max L for any given e
    pairs.sort()
    
    best_len = 0
    best_start = 1
    current_max_L = 0
    ptr = 0
    
    # Iterate through possible right endpoints e
    for e in range(1, N + 1):
        # Update current_max_L with all pairs that have R <= e
        while ptr < len(pairs) and pairs[ptr][0] <= e:
            L_val = pairs[ptr][1]
            if L_val > current_max_L:
                current_max_L = L_val
            ptr += 1
            
        # For this e, the minimum valid start is current_max_L + 1
        s = current_max_L + 1
        
        # Check if s <= e
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    if best_len == 0:
        # Fallback: should not happen, but just in case
        # Find any single element that is not self-inverse
        for i in range(1, N + 1):
            if inv[i] != i:
                return [i, 1]
        return [1, 1]
        
    return [best_start, best_len]
# EVOLVE-BLOCK-END
