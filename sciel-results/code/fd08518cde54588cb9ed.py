"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # We construct two disjoint K_{3,3} components.
    # K_{3,3} has vertices 0..5 and 6..11.
    # In K_{3,3}, all vertices have degree 3.
    # It is bipartite, so triangle-free.
    # The simple cycles in K_{3,3} have lengths 4 and 6.
    # Thus, the set of distinct cycle lengths is {4, 6}, size 2.
    # This is better than the seed program's score of 1.0 (value 5).
    # Score will be 5 / 2 = 2.5.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Part U = {0, 1, 2}, Part V = {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6..11
    # Part U = {6, 7, 8}, Part V = {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Convert to required format: list of [u, v] with int types
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
# EVOLVE-BLOCK-END
