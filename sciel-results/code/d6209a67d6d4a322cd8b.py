"""Rewritten program with optimized exhaustive search."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    n = P['n']
    min_v = P['minimum_vertices']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_vertices = list(range(n))
    
    best_score = 0
    best_subset = list(range(min_v)) # Fallback
    
    # Helper to count distinct degrees
    def get_distinct_degrees(subset):
        smask = 0
        for v in subset:
            smask |= (1 << v)
        degs = set()
        for v in subset:
            # Count neighbors in subset
            # smask has all vertices in subset. We want neighbors of v in subset, excluding v itself.
            # adj_mask[v] has all neighbors of v in the full graph.
            # adj_mask[v] & smask gives neighbors of v that are in subset.
            # Since v is in smask, and v is not a neighbor of itself (no self-loops), 
            # adj_mask[v] & smask is exactly the neighbors of v within the subset.
            deg = (adj_mask[v] & smask).bit_count()
            degs.add(deg)
        return len(degs)
    
    # Exhaustive search over all subsets of size >= min_v
    # Sizes 8 to 17
    for size in range(min_v, n + 1):
        if time.time() > deadline:
            break
            
        for subset in combinations(all_vertices, size):
            if time.time() > deadline:
                break
            
            score = get_distinct_degrees(subset)
            
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                
                # Theoretical max for this size is 'size'. 
                # If we reach 'size', we can't do better for this size.
                # But a larger size could have a higher score (e.g. size 9 with 9 distinct > size 8 with 8 distinct).
                # So we must continue to larger sizes.
                # However, if we find a subset of size 17 with 17 distinct degrees, that's the global max.
                # If we find a subset of size k with k distinct degrees, the score is k.
                # Since max score is n=17, if we find score 17, we can stop.
                if best_score == n:
                    return [int(x) for x in best_subset]

    return [int(x) for x in best_subset]
# EVOLVE-BLOCK-END
