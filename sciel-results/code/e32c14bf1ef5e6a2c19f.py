import time

def construct():
    # The Fano plane (PG(2,2)) is the smallest 3-uniform hypergraph with
    # chromatic number 3. It has exactly 7 edges on 7 vertices. Any
    # 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    # We embed it on vertices 0..6 and leave 7,8 isolated.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring: no edge is monochromatic.
    colors = [0, 1, 2, 1, 1, 2, 0, 0, 0]
    
    # Ensure all integers are Python int
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
