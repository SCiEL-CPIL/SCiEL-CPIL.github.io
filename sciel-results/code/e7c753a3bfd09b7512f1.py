"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency sets
_ADJ = [set() for _ in range(P['n'])]
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)

def _distinct_degrees(vertices):
    """Compute number of distinct degrees in induced subgraph on vertices."""
    vset = set(vertices)
    degs = set()
    for v in vset:
        degs.add(len(_ADJ[v] & vset))
    return len(degs)

def construct():
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    n = P['n']
    min_v = P['minimum_vertices']
    
    best_count = 0
    best_subset = tuple(range(min_v))
    
    # Strategy: exhaustive search over subsets of size 8-12, 
    # but with smart pruning and ordering
    
    # First, try all 8-vertex subsets (C(17,8) = 24310, very fast)
    for size in range(min_v, 13):
        if time.time() > deadline:
            break
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            d = _distinct_degrees(combo)
            if d > best_count:
                best_count = d
                best_subset = combo
                # If we've achieved max possible (size-1 distinct degrees), 
                # that's optimal for this size
                if best_count == size - 1:
                    # Could still try larger subsets, but check time
                    pass
    
    # Try some larger subsets if time permits
    if time.time() < deadline - 5:
        for size in range(13, n + 1):
            if time.time() > deadline:
                break
            for combo in combinations(range(n), size):
                if time.time() > deadline:
                    break
                d = _distinct_degrees(combo)
                if d > best_count:
                    best_count = d
                    best_subset = combo
    
    return list(best_subset)
# EVOLVE-BLOCK-END
