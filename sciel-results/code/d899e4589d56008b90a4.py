"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency structure for fast degree queries
_N = P['n']
_adj = [set() for _ in range(_N)]
for _a, _b in P['edges']:
    _adj[_a].add(_b)
    _adj[_b].add(_a)
_min_v = int(P['minimum_vertices'])
_deadline = time.time() + 70.0  # safety margin well under 90s wall-clock limit


def _distinct_degrees(subset):
    """Return the number of distinct degrees within the induced subgraph on `subset`."""
    s = set(subset)
    degs = set()
    for v in subset:
        d = 0
        nb = _adj[v]
        for w in nb:
            if w in s:
                d += 1
        degs.add(d)
    return len(degs)


def construct():
    global _deadline
    best_subset = list(range(_min_v))  # fallback: first 8 vertices
    best_score = _distinct_degrees(best_subset)

    # Enumerate every vertex subset of size >= minimum_vertices (all feasible, ~91k total).
    # Use bitmask-based enumeration for speed.
    # We'll use a more efficient approach: iterate over all bitmasks from min_v to n bits set.
    
    # Actually, let's use a smarter approach with early pruning.
    # The maximum possible distinct degrees for a subset of size k is k (all vertices have different degrees).
    # So the theoretical maximum is 17 (for the full graph).
    
    # Let's enumerate subsets by size, starting from larger sizes (more likely to have more distinct degrees)
    # and use early termination if we hit the theoretical max.
    
    all_vertices = tuple(range(_N))
    
    # Try sizes from largest to smallest
    for k in range(_N, _min_v - 1, -1):
        if time.time() > _deadline:
            break
        
        # Theoretical max for this size is k
        # If we already have best_score >= k, we can skip this size
        if best_score >= k:
            continue
            
        for combo in itertools.combinations(all_vertices, k):
            if time.time() > _deadline:
                break
            score = _distinct_degrees(combo)
            if score > best_score:
                best_score = score
                best_subset = list(combo)
                # If we've achieved the theoretical max for this size, we can stop
                # (but larger sizes might have been already checked, so we continue to smaller sizes)
                if best_score >= k:
                    # Can't do better than k for this size, but we're going to smaller sizes
                    # which have lower theoretical max, so we can break out of this size loop
                    break
    
    # Type-safe return: list of Python ints, length >= minimum_vertices
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
