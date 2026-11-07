"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency sets for fast degree computation
_ADJ = [set() for _ in range(P['n'])]
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)

def _distinct_degrees(subset):
    """Compute the number of distinct degrees in the induced subgraph on subset."""
    s = set(subset)
    degrees = []
    for v in subset:
        deg = sum(1 for u in s if u != v and u in _ADJ[v])
        degrees.append(deg)
    return len(set(degrees))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = P['n']
    min_vertices = P['minimum_vertices']
    
    best_count = 0
    best_subset = None
    
    # Enumerate all subsets of size >= 8
    # For n=17, total subsets of size >= 8 is C(17,8)+...+C(17,17) which is manageable
    for size in range(min_vertices, n + 1):
        if time.time() > deadline:
            break
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                break
            count = _distinct_degrees(subset)
            if count > best_count:
                best_count = count
                best_subset = subset
    
    if best_subset is None:
        # Fallback: return the first 8 vertices
        return list(range(min_vertices))
    
    return list(best_subset)
# EVOLVE-BLOCK-END
