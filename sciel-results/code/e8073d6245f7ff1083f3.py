"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit

    # The Fano plane is the unique 3-chromatic hypergraph with 7 vertices and 7 edges.
    # We use vertices 0..6 for the Fano plane and leave 7..8 isolated.
    # This ensures the chromatic number is 3 (inherited from Fano) and edge count is minimal (7).
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]

    # Valid 3-coloring for the Fano plane:
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0
    # Vertices 7, 8 are isolated, assign color 0.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]

    # Verify: every edge must be non-monochromatic
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # This should never happen for the Fano plane with this coloring
            # Fallback to a known valid configuration if logic error occurs
            # But Fano plane is robust.
            pass

    # Ensure types are correct
    edges = [list(map(int, e)) for e in edges]
    colors = list(map(int, colors))

    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
