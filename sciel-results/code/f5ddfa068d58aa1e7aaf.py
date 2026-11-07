"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The minimum number of edges for a 3-chromatic 3-uniform hypergraph is 7.
    # The Fano plane is the unique minimal example.
    # Vertices 0-6 form the Fano plane; 7-8 are isolated.
    
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
    # 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1
    # 7:0, 8:0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
