def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_length = 1
    
    for start in range(1, n + 1):
        # Try to extend interval from start
        # We need to check that for all x, y in [start, start+len-1], x*y % p != 1
        # Equivalent: for all x in interval, inv[x] not in interval
        
        # Use a set to track what's in the interval
        # Since interval is contiguous, we can check incrementally
        
        # For efficiency, note that if we add a new element x, we need inv[x] not to be in [start, x-1]
        # and also x not to be inv of any element in [start, x-1]
        
        max_len = 1
        # Check if start itself is self-inverse (inv[start] == start)
        if inv[start] == start:
            continue  # Can't include self-inverse elements
        
        # Try extending
        for end in range(start + 1, n + 1):
            x = end
            if inv[x] == x:
                break  # Self-inverse, can't include
            # Check if inv[x] is in [start, end-1]
            if start <= inv[x] <= end - 1:
                break
            # Check if x is inv of any element in [start, end-1]
            # This is equivalent to checking if any element y in [start, end-1] has inv[y] == x
            # Which means y = inv[x], but we already checked that inv[x] is not in [start, end-1]
            # So this is already covered!
            max_len = end - start + 1
        
        if max_len > best_length:
            best_length = max_len
            best_start = start
    
    return [best_start, best_length]
