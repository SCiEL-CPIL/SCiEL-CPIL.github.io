"""
Constructs a graph on vertices 0..11 with minimum degree 3, girth at least 4,
and exactly 2 distinct simple cycle lengths (4 and 6).

The graph consists of two disjoint copies of K_{3,3}.
- K_{3,3} is bipartite, so it is triangle-free (girth >= 4).
- The simple cycles in K_{3,3} have lengths 4 and 6.
- Since the components are disjoint, the set of cycle lengths for the whole graph is {4, 6}.
- The number of distinct cycle lengths is 2.
- It is a known lower bound that min-degree 3 triangle-free graphs cannot have only 1 distinct cycle length.
"""

import time
from collections import deque

_START_TIME = None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Safety deadline: 85 seconds to leave margin for evaluator overhead
    DEADLINE = _START_TIME + 85

    # We construct two disjoint K_{3,3} graphs.
    # Vertices 0-5 form the first K_{3,3}: A={0,1,2}, B={3,4,5}
    # Vertices 6-11 form the second K_{3,3}: A={6,7,8}, B={9,10,11}

    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]

    edges = []

    # Generate edges for first K_{3,3} component
    for u in part_a1:
        for v in part_b1:
            edges.append([int(u), int(v)])

    # Generate edges for second K_{3,3} component
    for u in part_a2:
        for v in part_b2:
            edges.append([int(u), int(v)])

    # Validation and cleanup: remove loops and duplicates, normalize ordering
    final_edges = []
    seen_pairs = set()

    for e in edges:
        u, v = e
        if u == v:
            continue
        pair_key = (min(u, v), max(u, v))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        final_edges.append([u, v])

    # Sort for deterministic output
    final_edges.sort()

    # Quick verification (lightweight, well within time budget):
    # Confirm min degree >= 3 and verify structural properties.
    # Since the construction is deterministic (two K_{3,3}), we can perform
    # strict checks without expensive enumeration.
    if time.time() < DEADLINE - 5:
        n = 12
        adj = [set() for _ in range(n)]
        for u, v in final_edges:
            adj[u].add(v)
            adj[v].add(u)

        # Check minimum degree
        min_deg = min(len(adj[i]) for i in range(n))
        assert min_deg >= 3, f"Min degree {min_deg} < 3"

        # Triangle-free check: no three mutually connected vertices
        # In a bipartite graph, this is guaranteed by construction, but we verify.
        # For K_{3,3}, parts are {0,1,2} and {3,4,5}. No edges within parts.
        # We verify no triangle exists.
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        raise ValueError("Triangle found, graph is not triangle-free")

        # Verify specific cycles exist to ensure girth is 4 and 6 are present.
        # 4-cycle: 0-3-1-4-0
        if not (3 in adj[0] and 1 in adj[3] and 4 in adj[1] and 0 in adj[4]):
            raise ValueError("Expected 4-cycle not found")
        
        # 6-cycle: 0-3-1-5-2-4-0
        if not (3 in adj[0] and 1 in adj[3] and 5 in adj[1] and 2 in adj[5] and 4 in adj[2] and 0 in adj[4]):
            raise ValueError("Expected 6-cycle not found")

        # Verify no other cycle lengths are possible in K_{3,3}
        # K_{3,3} is bipartite, so all cycles are even.
        # Max cycle length in 6-vertex graph is 6.
        # So lengths must be subset of {4, 6}.
        # We verified 4 and 6 exist.
        # Thus distinct lengths are exactly {4, 6}. Count = 2.

    return final_edges
