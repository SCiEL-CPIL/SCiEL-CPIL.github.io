"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.monotonic() + 80.0

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    min_size = P['minimum_vertices']

    best_deg = 0
    best_vertices = list(range(min_size))

    # Precompute for each vertex v, the bitmask of its neighbors
    # For a given subset mask, degree of v in induced subgraph = popcount(adj[v] & mask)
    
    # Enumerate all subsets with popcount >= min_size
    # 2^17 = 131072 total, manageable
    for mask in range(1 << n):
        if time.monotonic() > deadline:
            break
        pc = mask.bit_count()
        if pc < min_size:
            continue
        
        # Compute distinct degrees using a bitmask-based approach
        # For each vertex in the subset, compute its degree in the induced subgraph
        deg_mask = 0  # bitmask of degree values seen (degrees 0..16)
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            d = (adj[v] & mask).bit_count()
            deg_mask |= (1 << d)
            m ^= low
        
        nd = deg_mask.bit_count()
        if nd > best_deg:
            best_deg = nd
            # Extract vertices
            verts = []
            m2 = mask
            while m2:
                low = m2 & (-m2)
                v = low.bit_length() - 1
                verts.append(v)
                m2 ^= low
            best_vertices = verts
            # Early termination: max possible is min(pc, n) distinct degrees
            # If we found all possible, we can't do better
            if best_deg >= pc:
                break

    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
