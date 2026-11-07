import time

def construct():
    n = 17
    minv = 8
    edges = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], 
             [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], 
             [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], 
             [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], 
             [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], 
             [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], 
             [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], 
             [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], 
             [8, 9], [8, 10], [8, 12], [8, 16], 
             [9, 10], [9, 11], [9, 13], 
             [10, 11], [10, 12], [10, 14], 
             [11, 12], [11, 13], [11, 15], 
             [12, 13], [12, 14], [12, 16], 
             [13, 14], [13, 15], 
             [14, 15], [14, 16], 
             [15, 16]]

    # Build adjacency masks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    best_count = -1
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Iterate over all masks
    # We can iterate from 2^17 - 1 down to 2^8
    # Or just range(1 << n)
    
    # Optimization: Precompute bit_count for all possible masks?
    # 2^17 is 131072. A list of 131072 ints is fine.
    # popcount = [0] * (1 << n)
    # for i in range(1, 1 << n):
    #     popcount[i] = popcount[i >> 1] + (i & 1)
    # This might speed up the size check.
    
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Also precompute the list of vertices in each mask? No, that's memory heavy.
    # But we can precompute the degrees? No, degrees depend on the mask.
    
    # Let's stick to the inner loop but use the popcount array for size check.
    
    for mask in range(1 << n):
        # Check time every 1000 iterations to reduce overhead
        if mask % 1000 == 0 and time.time() > deadline:
            break
            
        sz = popcount[mask]
        if sz < minv:
            continue
            
        degs = set()
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            degs.add(deg)
            m ^= lb
            
        nd = len(degs)
        if nd > best_count:
            best_count = nd
            best_mask = mask
            # If we find a subset of size sz with sz distinct degrees, 
            # it's the max for this size. 
            # The global max is at most n.
            if best_count == n:
                break
            # Also, if best_count == sz, we can't improve for THIS mask, 
            # but a larger mask could have more distinct degrees.
            # However, max distinct degrees is bounded by the size.
            # If we find 8 distinct degrees in a size 8 mask, that's 8.
            # Can we find 9 in a size 9 mask? Yes.
            # So we must continue.

    if best_mask == 0:
        return [int(i) for i in range(minv)]
        
    verts = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        verts.append(int(v))
        m ^= lb

    return verts
