"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_min_vertices = P['minimum_vertices']

# Precompute adjacency as bitmask integers
_adj_mask = [0] * _N
for (u, v) in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

# Precompute adjacency matrix as numpy int8 array for vectorized ops
_adj_matrix = np.zeros((_N, _N), dtype=np.int8)
for (u, v) in P['edges']:
    _adj_matrix[u, v] = 1
    _adj_matrix[v, u] = 1

def _distinct_degrees_numpy(subset_indices):
    """Compute number of distinct degrees in induced subgraph using numpy vectorization."""
    k = len(subset_indices)
    # Build adjacency submatrix
    sub_adj = _adj_matrix[np.ix_(subset_indices, subset_indices)]
    # Degrees are row sums (diagonal is 0 since no self-loops)
    degrees = sub_adj.sum(axis=1)
    # Count distinct values
    return len(np.unique(degrees))

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin under 90s kill
    best_score = -1
    best_subset = None
    
    n = _N
    min_v = _min_vertices
    all_vertices = list(range(n))
    
    # Strategy: Use numpy vectorized evaluation for each subset size
    # Process larger subsets first since they can have more distinct degrees
    # For size k, max distinct degrees is k (degrees 0..k-1)
    # So we process k from n down to min_v
    
    for k in range(n, min_v - 1, -1):
        if time.time() > deadline:
            break
        
        # Upper bound for this k: at most k distinct degrees
        # If best_score >= k, no subset of size k can beat it
        if best_score >= k:
            continue
        
        # Enumerate all k-combinations
        # For k=17: 1, k=16: 17, k=15: 136, k=14: 680, k=13: 2380,
        # k=12: 6188, k=11: 12376, k=10: 19448, k=9: 24310, k=8: 24310
        # Total: ~90k combinations
        
        for combo in combinations(all_vertices, k):
            if time.time() > deadline:
                break
            
            score = _distinct_degrees_numpy(combo)
            if score > best_score:
                best_score = score
                best_subset = list(combo)
                if best_score >= k:
                    break  # Can't do better for this k
    
    if best_subset is None:
        best_subset = list(range(min_v))
    
    return best_subset
# EVOLVE-BLOCK-END
