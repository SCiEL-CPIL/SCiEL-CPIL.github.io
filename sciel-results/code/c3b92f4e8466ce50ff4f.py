"""
Constructs the minimal 3-uniform 3-chromatic hypergraph on 9 vertices.
The core structure is the Fano plane (7 vertices, 7 edges), which is the 
smallest 3-uniform hypergraph with chromatic number 3.
Vertices 7 and 8 are isolated.
"""
import time

def construct():
    """
    Returns a dict with keys 'edges' and 'colors'.
    'edges': List of 7 distinct triples on vertices 0..8.
    'colors': List of 9 integers in {0, 1, 2}.
    """
    # Deadline guard: 50s safety margin for 60s limit
    start_time = time.time()
    deadline = start_time + 50
    
    # Fano plane edges on vertices 0..6
    # These 7 edges form the Fano plane, which is 3-chromatic.
    # Vertices 7 and 8 are isolated.
    edges_raw = [
        (0, 1, 6),
        (0, 2, 5),
        (0, 3, 4),
        (1, 2, 4),
        (1, 3, 5),
        (2, 3, 6),
        (4, 5, 6)
    ]
    
    # Convert to list of lists of Python ints to satisfy type checks
    edges = [[int(x) for x in e] for e in edges_raw]
    
    # Valid 3-coloring for the Fano plane:
    # Color 0: vertices 0, 1, 2
    # Color 1: vertices 3, 4, 5
    # Color 2: vertex 6
    # Isolated vertices 7, 8 are assigned color 0.
    colors_raw = [0, 0, 0, 1, 1, 1, 2, 0, 0]
    colors = [int(c) for c in colors_raw]
    
    # Safety check (should never trigger for this static construction)
    if time.time() > deadline:
        # If we somehow exceeded time, return the valid static result immediately
        pass
    
    return {
        'edges': edges,
        'colors': colors
    }
