"""Rewritten program: vectorized bitmask search for max distinct degrees in induced subgraph."""
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _popcount(x):
    x = x.astype(np.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).astype(np.uint8)

def construct():
    n = int(P['n'])
    edges = P['edges']
    min_v = int(P['minimum_vertices'])
    
    # Precompute adjacency bitmasks for each vertex
    masks = np.zeros(n, dtype=np.uint32)
    for a, b in edges:
        masks[a] |= (1 << b)
        masks[b] |= (1 << a)

    # Generate all possible bitmasks
    # 2^17 = 131072
    all_masks = np.arange(1 << n, dtype=np.uint32)
    
    # Filter masks with popcount >= min_v
    popcounts = _popcount(all_masks)
    valid_indices = np.where(popcounts >= min_v)[0]
    
    if len(valid_indices) == 0:
        return list(range(min_v)) # Fallback, shouldn't happen
    
    valid_masks = all_masks[valid_indices]
    n_valid = len(valid_masks)
    
    # Compute degrees for all valid masks
    # deg_matrix[i, v] = degree of vertex v in subset valid_masks[i]
    deg_matrix = np.empty((n_valid, n), dtype=np.uint8)
    active = np.empty((n_valid, n), dtype=bool)
    
    for v in range(n):
        # Bitwise AND: mask & adj_mask[v] gives neighbors of v in the subset
        # Popcount gives the degree
        deg_matrix[:, v] = _popcount(valid_masks & masks[v])
        # Check if v is in the subset
        active[:, v] = ((valid_masks >> v) & 1) == 1

    # Count distinct degrees for each subset
    distinct_counts = np.zeros(n_valid, dtype=np.uint8)
    for d in range(n):
        # (deg_matrix == d) & active -> True if vertex v has degree d and is active
        # .any(axis=1) -> True if there is at least one active vertex with degree d
        has_degree_d = ((deg_matrix == d) & active).any(axis=1)
        distinct_counts += has_degree_d.astype(np.uint8)
        
    # Find the subset with max distinct degrees
    max_idx = int(np.argmax(distinct_counts))
    max_deg = int(distinct_counts[max_idx])
    
    # Convert winning bitmask to list of vertices
    winning_mask = int(valid_masks[max_idx])
    result = []
    for v in range(n):
        if (winning_mask >> v) & 1:
            result.append(v)
            
    return result
