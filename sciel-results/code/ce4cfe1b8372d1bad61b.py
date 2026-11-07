"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    added_edges = []
    
    # Strategy: Connect vertex 0 to all odd vertices (B partition) except those already connected.
    # This creates a bipartite graph with diameter 4.
    # Vertices: 0..23.
    # A = {0, 2, ..., 22}, B = {1, 3, ..., 23}.
    # Add (0, b) for b in B if (0, b) not in base.
    
    for b in range(1, n, 2):
        if (0, b) not in base_edges:
            added_edges.append([0, b])
            
    return added_edges
# EVOLVE-BLOCK-END
