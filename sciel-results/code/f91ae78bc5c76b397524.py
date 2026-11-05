"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses in [1, 1008]
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x where x*x % p == 1
    # These are x=1 and x=p-1=1008
    self_inv = {1, 1008}
    
    # Two-pointer sliding window approach (O(N))
    # Maintain window [left, right] that is valid.
    # For each right, advance left as needed to maintain validity.
    
    best_len = 0
    best_start = 1
    left = 1
    
    for right in range(1, n + 1):
        # If right is self-inverse, we can't include it
        if right in self_inv:
            left = right + 1
            continue
        
        # If inv[right] is in [left, right-1], we must exclude it
        ie = inv[right]
        if left <= ie < right:
            left = ie + 1
        
        # Now [left, right] should be valid
        if left <= right:
            length = right - left + 1
            if length > best_len:
                best_len = length
                best_start = left
    
    if best_len == 0:
        return [2, 1]
    
    # Verify the result
    s = best_start
    e = s + best_len - 1
    interval_set = set(range(s, e + 1))
    for x in interval_set:
        if (x * x) % p == 1:
            return [2, 1]
        if inv[x] in interval_set:
            return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
