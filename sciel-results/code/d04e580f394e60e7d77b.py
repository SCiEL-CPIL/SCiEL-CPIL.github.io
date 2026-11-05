def construct():
    MOD = 1009
    N = 1008
    
    # Precompute inverses and representatives
    inv = [0] * (MOD + 1)
    rep = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        if x == 1 or x == 1008:
            rep[x] = x
        else:
            rep[x] = min(x, inv[x])
            
    # Sliding window
    counts = [0] * (MOD + 1)
    L = 1
    best_len = 0
    best_start = 1
    
    for R in range(1, MOD):
        # Handle self-inverse: cannot be in window
        if R in (1, 1008):
            # Clear window
            while L < R:
                counts[rep[L]] -= 1
                L += 1
            L = R + 1
            continue
        
        r_rep = rep[R]
        
        # If r_rep is already in window, move L until it's removed
        while counts[r_rep] > 0:
            counts[rep[L]] -= 1
            L += 1
            
        # Add R to window
        counts[r_rep] += 1
        
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
            
    return [best_start, best_len]
