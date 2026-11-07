"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """
    Returns the Fano plane embedded in 9 vertices.
    The Fano plane has 7 edges and is the smallest 3-uniform hypergraph
    that is not 2-colorable (i.e., it is 3-chromatic).
    Since 7 is the theoretical minimum for 3-chromatic 3-uniform hypergraphs,
    this is the optimal solution.
    """
    # Fano plane edges on vertices 0-6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6:
    # 0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 0, 6: 1
    # Vertices 7 and 8 are isolated, assign color 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
