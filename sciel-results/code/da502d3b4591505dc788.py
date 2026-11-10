"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute modular inverses for 1..1008
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For each start, find the maximum end such that [start, end] is valid
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is NOT in [s, e]
    
    best_start = 1
    best_len = 0
    
    n = p - 1  # 1008
    
    for s in range(1, n + 1):
        # Try to extend from s
        e = s
        valid = True
        while e <= n and valid:
            # Check if adding e to the interval breaks validity
            # We need: for all x in [s, e], inv[x] not in [s, e]
            # When we extend to e, we need to check:
            # 1. inv[e] not in [s, e]
            # 2. For all x in [s, e-1], inv[x] != e (i.e., e not in [s, e])
            #    This is equivalent to: inv[e] not in [s, e] AND no x in [s,e-1] has inv[x] == e
            #    But inv[e] == x iff inv[x] == e, so condition is: inv[e] not in [s, e]
            #    Wait, that's not quite right. Let me reconsider.
            
            # Actually, the condition for [s, e] to be valid is:
            # For all x in [s, e], inv[x] not in [s, e]
            # This is equivalent to: the set [s, e] does not contain any pair {x, inv[x]}
            
            # When extending from [s, e-1] to [s, e], the new constraints are:
            # - inv[e] must not be in [s, e]
            # - For each x in [s, e-1], inv[x] must not equal e
            #   i.e., e must not be the inverse of any x in [s, e-1]
            #   i.e., inv[e] must not be in [s, e-1] (since inv is an involution)
            # Combined: inv[e] not in [s, e]
            
            ie = inv[e]
            if s <= ie <= e:
                valid = False
            else:
                e += 1
        
        # Interval [s, e-1] is the longest valid interval starting at s
        length = (e - 1) - s + 1 if e > s else 0
        if e > s:
            length = e - s
        else:
            length = 0
        
        if length > best_len:
            best_len = length
            best_start = s
    
    if best_len == 0:
        # Fallback: single element that is not self-inverse
        # 1 is self-inverse, 1008 is self-inverse
        # Any other single element is valid
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
