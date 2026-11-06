"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return a C5-free graph with no independent set of size 9.
    We use a construction based on 8 disjoint K4 components plus carefully
    designed additional vertices to push beyond 32.

    Key insight: The 8xK4 construction gives n=32 with alpha=8.
    To exceed 32, we need to add vertices while maintaining:
    - No 5-cycles
    - No independent set of size 9

    Strategy: Add vertices that connect to multiple K4s in a way that
    prevents forming independent sets of size 9, while avoiding 5-cycles.

    A known approach: Use a "blow-up" or add a central vertex connected
    to one vertex in each K4, but this creates alpha issues.

    Better: Add 2 vertices that form an edge, and connect each to 4 different
    K4s (one vertex per K4) such that no 5-cycle is formed.

    However, the simplest guaranteed improvement: Use 8xK4 + 1 vertex.
    The 9th vertex must connect to at least one vertex in each K4 to prevent
    picking it plus 8 others (one from each K4) as an independent set.

    If we connect vertex 32 to vertex 0 of each K4 (vertices 0,4,8,...,28),
    then any independent set can contain at most one vertex from each K4,
    and if it contains vertex 32, it cannot contain any of 0,4,8,...,28.
    So max independent set size is still 8 (either pick 32 and 7 from K4s,
    or pick 8 from K4s excluding the connected ones).

    Wait, that's not right. Let me think more carefully.

    Actually, the known best is 32. Let me just implement the 8xK4 which is
    known to be valid and gives n=32.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
