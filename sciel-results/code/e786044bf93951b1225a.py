"""Program: minimize distinct simple cycle lengths on 12 vertices with min degree 3, girth >= 4."""

import time

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) as two disjoint K_3,3 components.

    K_3,3 properties:
    - Simple, undirected, bipartite
    - Minimum degree = 3 (every vertex connects to all 3 vertices on the other side)
    - Girth = 4 (no triangles since bipartite)
    - Cycle lengths present: 4 and 6 only
      * 4-cycles: choose 2 left + 2 right vertices -> C4
      * 6-cycles: use all 6 vertices in alternating pattern -> C6
      * No other lengths possible in K_3,3

    Two disjoint K_3,3 copies: same cycle lengths {4, 6}, count = 2.

    This is optimal: value 1 is impossible because any min-degree-3 triangle-free
    graph (connected or not) with 12 vertices must have at least two distinct
    cycle lengths. Each component needs >= 6 vertices for triangle-free min-degree-3,
    so max 2 components, each contributing {4,6}.
    """
    t0 = time.time()
    deadline = t0 + 5.0  # generous safety margin; this is a static construction

    # Component 1: K_3,3 on vertices {0,1,2} x {3,4,5}
    # Component 2: K_3,3 on vertices {6,7,8} x {9,10,11}
    edges = []
    for base in (0, 6):
        left = [base, base + 1, base + 2]
        right = [base + 3, base + 4, base + 5]
        for u in left:
            for v in right:
                edges.append([u, v])

    # Type safety: ensure all elements are Python int, edges are lists
    result = [[int(u), int(v)] for u, v in edges]

    # Verify: 18 edges, 12 vertices, no loops, no duplicates
    assert len(result) == 18, f"Expected 18 edges, got {len(result)}"
    assert all(isinstance(e, list) and len(e) == 2 for e in result)
    assert all(isinstance(x, int) and not isinstance(x, bool) for e in result for x in e)
    assert all(e[0] != e[1] for e in result), "Loop detected"
    edge_set = set()
    for u, v in result:
        key = (min(u, v), max(u, v))
        assert key not in edge_set, f"Duplicate edge {key}"
        edge_set.add(key)
    assert all(0 <= u <= 11 and 0 <= v <= 11 for u, v in result)

    # Deadline guard
    if time.time() > deadline:
        return result

    return result

# EVOLVE-BLOCK-END
