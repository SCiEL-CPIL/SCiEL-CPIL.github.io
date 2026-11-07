"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_size = P['minimum_vertices']
    deadline = time.monotonic() + 85.0

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    best_deg = 0
    best_vertices = list(range(min_size))

    # Iterate over all subsets using bitmask enumeration
    # 2^17 = 131072 subsets, which is very fast
    for mask in range(1 << n):
        if time.monotonic() > deadline:
            break
            
        size = mask.bit_count()
        if size < min_size:
            continue
        
        # Pruning: if the subset size is not greater than the current best distinct degrees,
        # it cannot improve the score (max distinct degrees is size).
        if size <= best_deg:
            continue

        degs = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            degs.add((adj[v] & mask).bit_count())
            m ^= lsb
            
        nd = len(degs)
        if nd > best_deg:
            best_deg = nd
            verts = []
            m = mask
            while m:
                lsb = m & -m
                verts.append(lsb.bit_length() - 1)
                m ^= lsb
            best_vertices = verts
            
            # If we achieved the maximum possible distinct degrees for the full graph size,
            # or if we found a subset where distinct degrees == size (perfect), 
            # we might still find a larger subset with more distinct degrees.
            # But if best_deg == n, we can stop.
            if best_deg == n:
                break

    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
