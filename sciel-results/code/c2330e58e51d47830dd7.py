def construct():
    import time
    deadline = time.time() + 85
    
    n = 1008
    p = 1009
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    best_start = 1
    best_length = 0
    
    # Brute force O(N^2) search
    # For each start, find the maximum end such that [start, end] is valid.
    for s in range(1, n + 1):
        # Check if s itself is a self-inverse
        if inv[s] == s:
            continue # Length 0, skip
        
        max_e = s - 1
        for e in range(s, n + 1):
            # If e is self-inverse, invalid
            if inv[e] == e:
                break
            # If inv[e] is in [s, e-1], invalid
            if s <= inv[e] < e:
                break
            # Otherwise, [s, e] is valid so far
            max_e = e
            
        length = max_e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
            
        # Early termination if time is running out
        if time.time() > deadline:
            break

    return [best_start, best_length]
