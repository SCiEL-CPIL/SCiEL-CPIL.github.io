"""Rewritten program: directly returns the known optimal 7-edge Fano plane construction."""

def construct():
    # Fano plane on vertices 0-6 (7 edges), vertices 7-8 isolated
    # This is the minimum non-2-colorable 3-uniform hypergraph
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    # Valid 3-coloring: no edge is monochromatic
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}
