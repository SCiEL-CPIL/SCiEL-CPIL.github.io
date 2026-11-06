def construct():
    import time
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 1008
    p = 1009
    
    # Precompute modular inverses mod 1009 for x in [1, 1008]
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    # min_v[u] will store the minimum v such that there is a pair (u, v) 
    # where u = min(x, inv[x]) and v = max(x, inv[x])
    # We want to find for each start s, the minimum v among all pairs with u >= s.
    
    # Initialize min_v with a large number (infinity proxy)
    INF = n + 1
    min_v = [INF] * (n + 2)
    
    for x in range(1, n + 1):
        ix = inv[x]
        u = x if x < ix else ix
        v = ix if x < ix else x
        if v < min_v[u]:
            min_v[u] = v
            
    # Compute suffix minimums of min_v
    # suffix_min_v[s] = min(min_v[s], suffix_min_v[s+1])
    suffix_min_v = [INF] * (n + 2)
    suffix_min_v[n + 1] = INF
    for s in range(n, 0, -1):
        if min_v[s] < suffix_min_v[s + 1]:
            suffix_min_v[s] = min_v[s]
        else:
            suffix_min_v[s] = suffix_min_v[s + 1]
            
    best_start = 1
    best_length = 0
    
    for s in range(1, n + 1):
        # e_max is the largest e such that e < min_v for all pairs with u >= s
        # min_v for u >= s is suffix_min_v[s]
        e_max = suffix_min_v[s] - 1
        
        if e_max >= s:
            length = e_max - s + 1
            if length > best_length:
                best_length = length
                best_start = s
                
    return [best_start, best_length]
