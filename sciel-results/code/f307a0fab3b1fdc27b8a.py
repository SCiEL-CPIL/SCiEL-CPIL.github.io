"""Constructs a graph with minimum degree 3, girth 4, and minimal distinct cycle lengths."""

import time

def construct():
    # Deadline guard: return immediately if we're close to the 90s limit
    _t0 = time.time()
    
    # The goal is to minimize the number of distinct simple cycle lengths.
    # A graph formed by the disjoint union of two K_{3,3} graphs satisfies all constraints:
    # 1. Vertices 0-11 are used.
    # 2. K_{3,3} is 3-regular, so every vertex has degree 3 (>= 3).
    # 3. K_{3,3} is bipartite, so it is triangle-free (girth >= 4).
    # 4. K_{3,3} contains cycles (4-cycles and 6-cycles).
    # 5. The distinct cycle lengths in K_{3,3} are exactly {4, 6}.
    #    - 4-cycles exist (e.g., 0-3-1-4-0).
    #    - 6-cycles exist (e.g., 0-3-1-4-2-5-0).
    #    - No cycles of length > 6 exist because the partition size is 3.
    #    - No odd cycles exist because it is bipartite.
    # Thus, the set of distinct cycle lengths is {4, 6}, which has size 2.
    # This is the minimum possible for n=12 and min_degree>=3.
    # Any attempt to reduce this to 1 distinct cycle length (e.g., only 4s) fails
    # because 3-regular bipartite graphs on 12 vertices (like two K3,3 or Q3 + 4 vertices)
    # inevitably contain 6-cycles or require degrees < 3.
    
    edges = []
    
    # First component: K_{3,3} on vertices 0, 1, 2 and 3, 4, 5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Second component: K_{3,3} on vertices 6, 7, 8 and 9, 10, 11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    # Safety check: ensure we haven't exceeded time budget
    if time.time() - _t0 > 85.0:
        # Should not happen for this O(1) construction, but guard just in case
        pass
    
    # Ensure types are strictly Python int and list
    # (They already are, but explicit conversion ensures compliance with strict type checks)
    return [[int(u), int(v)] for u, v in edges]
