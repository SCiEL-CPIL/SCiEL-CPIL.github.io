"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 60.0

    # Structural change: Construct the graph using a different method.
    # We still construct two disjoint K_{3,3} components to achieve the minimum
    # distinct cycle lengths of 2 (lengths 4 and 6).
    # Instead of nested loops, we define the edges explicitly or via a helper.
    
    edges = []
    
    # Component 1: Vertices 0-5
    # Bipartite sets A1={0,1,2}, B1={3,4,5}
    # All cross edges exist.
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: Vertices 6-11
    # Bipartite sets A2={6,7,8}, B2={9,10,11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])

    # Ensure types are int
    result = [[int(u), int(v)] for u, v in edges]
    
    return result
# EVOLVE-BLOCK-END
