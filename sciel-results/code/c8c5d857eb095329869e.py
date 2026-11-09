"""Find the maximum interval in [1,1008] with no modular inverse pair (including self-inverse)."""
# EVOLVE-BLOCK-START
def construct():
    mod = 1009
    
    # Precompute modular inverses
    inv = [0] * (mod + 1)
    for x in range(1, mod):
        inv[x] = pow(x, mod - 2, mod)
        
    # For each x in [2, 1007], if inv[x] > x, we have a pair (x, inv[x]).
    # We need the longest interval [a, b] in [2, 1007] that does not contain both elements of any pair.
    # For a start a, the max end b is limited by the smallest y in any pair (x, y) with x >= a.
    # Let min_y_ge[a] = min{ y | (x, y) is a pair, x >= a }
    # Then max valid b = min_y_ge[a] - 1
    
    # Initialize min_y_ge to a value larger than 1007
    min_y_ge = [1009] * (mod + 1)
    
    for x in range(2, 1008):
        y = inv[x]
        if y > x:
            min_y_ge[x] = min(min_y_ge[x], y)
            
    # Compute suffix minimums
    suffix_min = [1009] * (mod + 1)
    suffix_min[1008] = 1009
    for i in range(1007, 1, -1):
        suffix_min[i] = min(min_y_ge[i], suffix_min[i+1])
        
    best_start = 2
    best_len = 0
    
    for a in range(2, 1008):
        b_max = suffix_min[a] - 1
        if b_max > 1007:
            b_max = 1007
        if b_max >= a:
            length = b_max - a + 1
            if length > best_len:
                best_len = length
                best_start = a
                
    if best_len == 0:
        return [2, 1]
        
    return [best_start, best_len]
# EVOLVE-BLOCK-END
