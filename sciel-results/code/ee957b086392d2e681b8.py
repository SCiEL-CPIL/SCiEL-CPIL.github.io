def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_length = 0
    
    # For each start s, compute max valid end
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        
        max_e = n  # 1008
        
        for x in range(s, n + 1):
            ix = inv[x]
            if ix >= s:
                # Conflict when e >= max(x, ix)
                conflict = max(x, ix)
                if conflict - 1 < max_e:
                    max_e = conflict - 1
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    # Verify the result
    s = best_start
    e = s + best_length - 1
    for x in range(s, e + 1):
        for y in range(s, e + 1):
            if (x * y) % p == 1:
                # This shouldn't happen; fall back to a safe answer
                return [2, 1]
    
    return [best_start, best_length]
