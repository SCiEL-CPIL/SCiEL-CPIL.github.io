import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    n = P['n']
    edges = P['edges']
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    is_valid = [False] * (1 << n)
    
    for mask in range(1, 1 << n):
        is_c = True
        is_i = True
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if (adj[v] & mask) != (mask ^ lsb):
                is_c = False
            if (adj[v] & mask) != 0:
                is_i = False
            if not is_c and not is_i:
                break
            m ^= lsb
            
        if is_c or is_i:
            is_valid[mask] = True
            
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
            
    # If deadline hit, we might have incomplete dp. 
    # But for n=14, it finishes instantly. 
    # If it doesn't, we fallback to greedy? 
    # The prompt says "return the best result found". 
    # If dp is incomplete, parent might be -1. 
    # However, n=14 is small enough that this won't happen.
    
    parts = []
    cur = full_mask
    while cur:
        sub = parent[cur]
        if sub == -1:
            # Fallback: add remaining as singletons? Or just break?
            # This shouldn't happen.
            break
        part = []
        m = sub
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            part.append(v)
            m ^= lsb
        parts.append(part)
        cur ^= sub
        
    # Ensure all vertices are covered if we broke early?
    # For n=14, we assume completion.
    
    return parts
