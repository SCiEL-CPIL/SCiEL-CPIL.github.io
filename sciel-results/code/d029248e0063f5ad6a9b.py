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

    start_time = time.time()
    deadline = start_time + 80.0

    # Build adjacency as list of neighbor lists and masks
    adj_list = [[] for _ in range(n)]
    adj_mask = [0] * n
    for a, b in edges:
        adj_list[a].append(b)
        adj_list[b].append(a)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)

    # Precompute for each vertex: mask of neighbors
    # Also precompute degree in full graph
    full_deg = [len(adj_list[i]) for i in range(n)]

    best_count = -1
    best_mask = 0
    best_verts = []

    # For each possible subset size from n down to minv, search for max distinct degrees
    # Actually, we want to maximize distinct degrees, so larger subsets might help.
    # But the constraint is at least 8.
    # Let's iterate over all masks again, but with a tighter inner loop.
    
    # Optimization: iterate masks by popcount to potentially find large subsets first?
    # No, just iterate all. 131k is tiny.
    
    # However, we can skip masks that are subsets of known better masks? No.
    
    # Let's just do the exhaustive search but make the inner loop as fast as possible.
    # The previous program used bit_count on (adj[v] & mask).
    # We can precompute adj_mask for each v.
    
    for mask in range(1 << n):
        if time.time() > deadline:
            break
        
        # Quick size check
        # bit_count is fast
        sz = mask.bit_count()
        if sz < minv:
            continue
        
        # Compute distinct degrees
        # We can use a local variable for speed
        degs = set()
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj_mask[v] & mask).bit_count()
            degs.add(deg)
            m ^= lb
            
        nd = len(degs)
        if nd > best_count:
            best_count = nd
            best_mask = mask
            # If we hit the theoretical max for this size?
            # Max distinct degrees in a graph of size sz is sz.
            # But degrees are bounded by sz-1.
            # If nd == sz, we can't do better for this sz.
            # But a larger sz could have more.
            # Max possible is 9 (since graph is 8-regular, max degree 8, min 0).
            # If nd == 9, we can stop?
            # Yes, 9 is the absolute maximum for this graph.
            if nd == 9:
                break

    # Extract vertices
    if best_mask == 0:
        # Fallback: pick first 8 vertices
        verts = list(range(8))
    else:
        verts = []
        m = best_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(int(v))
            m ^= lb

    return verts
