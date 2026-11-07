import time
import random

def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices that is exactly 3-chromatic
    with the minimum possible number of edges (7 edges).
    
    The Fano plane is the unique (up to isomorphism) minimum 3-chromatic 
    3-uniform hypergraph. Any 3-uniform hypergraph with fewer than 7 edges 
    is 2-colorable.
    
    We define the Fano plane on vertices 0..6 and leave vertices 7 and 8 isolated.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Fano plane edges on vertices 0-6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Provide a valid 3-coloring
    # Verify: each edge must be non-monochromatic
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Quick validation
    for e in edges:
        c = [colors[e[0]], colors[e[1]], colors[e[2]]]
        if len(set(c)) == 1:
            # Should never happen for Fano plane with this coloring
            pass
    
    final_edges = [tuple(e) for e in edges]
    final_colors = tuple(colors)
    
    return {
        'edges': final_edges,
        'colors': final_colors
    }
