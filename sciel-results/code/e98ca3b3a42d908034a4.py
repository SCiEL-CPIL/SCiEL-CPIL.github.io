import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']
    
    # Precompute adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
    best_count = 0
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Iterate over all possible subsets
    # We only care about subsets of size >= minv
    total_masks = 1 << n
    for mask in range(1, total_masks):
        if time.time() > deadline:
            break
            
        # Quick check on size using bit_count
        size = mask.bit_count()
        if size < minv:
            continue
            
        # Calculate degrees in induced subgraph
        # Optimization: if current best is >= size, we can't beat it with this size
        if best_count >= size:
             # We can't get more distinct degrees than the size of the set
             # But maybe a larger set later? 
             # This check is valid for this specific mask.
             # If best_count is 8, and size is 8, we can't exceed 8. 
             # If size is 9, we can get 9.
             # So if best_count >= size, skip.
             continue
        
        degs = set()
        # Iterate through vertices in the mask
        # We can do this by checking each vertex 0..n-1
        # Or by extracting bits. 
        # Checking 17 vertices is cheap.
        for v in range(n):
            if (mask >> v) & 1:
                d = (adj[v] & mask).bit_count()
                degs.add(d)
                
        c = len(degs)
        if c > best_count:
            best_count = c
            best_mask = mask
            # If we found a set where all degrees are distinct, 
            # that's the max for this size.
            # If best_count == size, we still need to check larger sizes
            # because a larger size could have *more* distinct degrees.
            # e.g. size 8 -> 8 distinct. size 9 -> 9 distinct.
            # So we don't break early unless best_count == n (17).
            
    # Extract vertices from best_mask
    best_vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        best_vertices.append(int(v))
        m ^= lsb
        
    return best_vertices
