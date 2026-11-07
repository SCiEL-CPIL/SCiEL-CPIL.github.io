import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _distinct_degrees(subset, adj_mask):
    """Compute the number of distinct degrees in the induced subgraph on subset."""
    smask = 0
    for v in subset:
        smask |= (1 << v)
    deg_set = set()
    for v in subset:
        deg = bin(adj_mask[v] & smask).count('1')
        deg_set.add(deg)
    return len(deg_set), smask

def _local_improve(subset, adj_mask, n, deadline):
    """Try single-vertex swaps to increase distinct degrees."""
    best = list(subset)
    best_nd, _ = _distinct_degrees(best, adj_mask)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, v in enumerate(best):
            for w in range(n):
                if w not in best:
                    candidate = list(best)
                    candidate[i] = w
                    nd, _ = _distinct_degrees(candidate, adj_mask)
                    if nd > best_nd:
                        best = candidate
                        best_nd = nd
                        improved = True
    return best, best_nd

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset = list(range(8))
    best_distinct = 0
    
    start = time.time()
    deadline = start + 80.0
    
    # Phase 1: Exhaustive search over subset sizes
    for size in range(8, n + 1):
        for subset in combinations(range(n), size):
            # Deadline check
            if time.time() > deadline:
                break
            
            # Build subset bitmask and compute distinct degrees
            nd, _ = _distinct_degrees(subset, adj_mask)
            
            if nd > best_distinct:
                best_distinct = nd
                best_subset = list(subset)
        
        if time.time() > deadline:
            break
    
    # Phase 2: Local improvement via single-vertex swaps
    best_subset, best_distinct = _local_improve(best_subset, adj_mask, n, deadline)
    
    # Phase 3: If time remains, try local improvement from other promising subsets
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        # Try a few more starting points from larger subsets
        for size in [n, n - 1, n - 2]:
            if time.time() > deadline:
                break
            for subset in combinations(range(n), size):
                if time.time() > deadline:
                    break
                nd, _ = _distinct_degrees(subset, adj_mask)
                if nd >= best_distinct:
                    improved, improved_nd = _local_improve(list(subset), adj_mask, n, deadline)
                    if improved_nd > best_distinct:
                        best_distinct = improved_nd
                        best_subset = improved
                if time.time() > deadline:
                    break
    
    return [int(v) for v in best_subset]
