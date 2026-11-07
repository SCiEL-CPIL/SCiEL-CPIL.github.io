"""Optimized search for maximum distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute degree for each vertex in each mask
    # deg[v][mask] = number of neighbors of v that are in mask
    total_masks = 1 << n
    deg = [[0] * total_masks for _ in range(n)]
    
    for v in range(n):
        dv = deg[v]
        av = adj[v]
        for mask in range(1, total_masks):
            lsb = mask & -mask
            prev_mask = mask ^ lsb
            u = lsb.bit_length() - 1
            if av & (1 << u):
                dv[mask] = dv[prev_mask] + 1
            else:
                dv[mask] = dv[prev_mask]

    best_count = -1
    best_mask = 0

    # Search strategy: try larger subsets first (they can have more distinct degrees)
    # Then smaller ones. Use combinatorial generation by size.
    
    # Try all subset sizes from n down to minv
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
        
        # The maximum possible distinct degrees for a subset of size 'size' is 'size'
        # If we already have best_count >= size, no subset of this size can beat it
        if best_count >= size:
            continue
        
        # Generate all combinations of 'size' vertices
        for combo in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            
            # Build mask from combo
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Count distinct degrees using precomputed table
            degs = set()
            for v in combo:
                degs.add(deg[v][mask])
            
            c = len(degs)
            if c > best_count:
                best_count = c
                best_mask = mask
                # Can't do better than 'size' for this size
                if best_count == size:
                    break
        
        if best_count == n:
            break

    # Extract vertices from best_mask
    best_vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        best_vertices.append(int(v))
        m ^= lsb

    return best_vertices
# EVOLVE-BLOCK-END
