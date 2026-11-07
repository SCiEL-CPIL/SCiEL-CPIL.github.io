"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def construct():
    n = P['n']
    edges = []
    
    # Construct two disjoint K_{3,3} components.
    # Component 1: Vertices 0, 1, 2 connected to 3, 4, 5.
    # Component 2: Vertices 6, 7, 8 connected to 9, 10, 11.
    
    # K_{3,3} on vertices 0-5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # K_{3,3} on vertices 6-11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
