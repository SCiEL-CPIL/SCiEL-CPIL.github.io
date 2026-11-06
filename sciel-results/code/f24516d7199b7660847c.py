import time

def construct():
    p = 1009
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    # We search for the maximum length L
    # L can be at most 1008
    # We iterate L from 1008 down to 1
    # For each L, we iterate start s from 1 to 1009 - L
    # If we find a valid interval, we return it immediately as it's the max L
    
    deadline = time.time() + 55  # Safety margin
    
    for L in range(1008, 0, -1):
        if time.time() > deadline:
            # Fallback to a known valid small interval if we time out (unlikely)
            return [2, 1]
            
        max_start = 1009 - L
        for s in range(1, max_start + 1):
            e = s + L - 1
            is_valid = True
            # Check all x in [s, e]
            for x in range(s, e + 1):
                ix = inv[x]
                if s <= ix <= e:
                    is_valid = False
                    break
            if is_valid:
                return [s, L]
                
    # Should never reach here, but return a valid default
    return [2, 1]
