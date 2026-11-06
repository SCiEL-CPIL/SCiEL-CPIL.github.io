import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # Self-inverse elements: x^2 = 1 mod 1009 => x = 1 or x = 1008
    self_inv = {1, 1008}
    
    best_len = 0
    best_start = 2 # Default valid start
    
    # Iterate over all possible starting positions
    for s in range(1, N + 1):
        # If s is self-inverse, the interval [s, ...] is immediately invalid
        if s in self_inv:
            continue
            
        # Optimization: if the max possible length from s is not better, skip
        if N - s + 1 <= best_len:
            continue
            
        current_end = s - 1
        
        # Expand interval to the right
        for e in range(s, N + 1):
            # Check if e itself is self-inverse
            if e in self_inv:
                break
                
            # Check if the inverse of e is already in the interval [s, e-1]
            # If inv[e] is in [s, e-1], then adding e creates a pair (inv[e], e)
            # Note: inv[e] cannot be e here because we checked self_inv.
            # So we just check if s <= inv[e] < e
            if s <= inv[e] < e:
                break
                
            # If we reach here, [s, e] is valid
            current_end = e
            
        length = current_end - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    if best_len == 0:
        return [2, 1]
        
    return [best_start, best_len]
