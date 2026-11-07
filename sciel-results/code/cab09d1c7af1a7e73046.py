"""Optimal 3-chromatic 3-uniform hypergraph (Fano Plane core)."""

def construct():
    # The Fano plane is the unique minimal 3-chromatic 3-uniform hypergraph.
    # It has 7 edges. Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    # We embed it on vertices 0-6. Vertices 7-8 are isolated and do not affect chromatic number.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring:
    # 0:0, 1:1, 2:2
    # 3:1, 4:2, 5:0
    # 6:1
    # 7:0, 8:0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}
