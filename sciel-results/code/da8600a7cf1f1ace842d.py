"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique 3-chromatic 3-uniform hypergraph with 7 edges.
    # We use vertices 0..6 for the Fano plane and leave 7..8 isolated.
    # This ensures the chromatic number is 3 and edge count is minimal (7).
    
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

    # Ensure types are correct (Python int)
    edges = [list(map(int, e)) for e in edges]
    colors = list(map(int, colors))

    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
