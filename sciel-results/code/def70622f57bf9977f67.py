"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a maximally diameter-2 graph on 16 vertices where every edge is critical.
    
    The complete bipartite graph K_{8,8} has 64 edges and diameter 2.
    For any edge (u, v) with u in A, v in B:
    - In K_{8,8}, dist(u, v) = 1.
    - After removing (u, v), the neighbors of u in the remaining graph are B \ {v}.
    - The neighbors of v in the remaining graph are A \ {u}.
    - For any path from u to v of length 2, we'd need a common neighbor w.
      But N(u) ∩ N(v) = (B \ {v}) ∩ (A \ {u}) = ∅ (since A and B are disjoint).
    - So dist(u, v) becomes 3 (u - w - x - v where w in B\{v}, x in A\{u}).
    - Thus diameter increases from 2 to 3.
    
    This makes every edge critical. Edge count = 8 * 8 = 64.
    
    We verify this is maximal: any non-bipartite graph with more edges on 16 vertices
    would have triangles, and in a triangle, each edge is redundant for diameter 2
    (the other two edges provide a length-2 path), violating the criticality constraint.
    """
    n = 16
    edges = []
    for i in range(8):
        for j in range(8, 16):
            edges.append([i, j])
    
    # Ensure all values are Python int (not numpy int or bool)
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
