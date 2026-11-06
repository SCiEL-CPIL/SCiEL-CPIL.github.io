"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n = 20
    left = list(range(10))
    right = list(range(10, 20))
    m = 10
    K = 20  # number of available colours

    # We will construct a C4-free bipartite graph.
    # This ensures no 4-cycles exist, so the rainbow condition is vacuously true.
    # We maximize edges by greedily adding edges that do not create a C4.
    
    # Adjacency lists
    # adjL[u] = set of v in right connected to u
    # adjR[v] = set of u in left connected to v
    adjL = [set() for _ in range(m)]
    adjR = [set() for _ in range(m)]
    
    edges = []
    
    # List of all potential edges
    all_edges = [(i, j) for i in range(m) for j in range(m)]
    random.shuffle(all_edges)
    
    for i, j in all_edges:
        if time.time() > deadline:
            break
        
        # Check if adding (i, j) creates a C4.
        # A C4 is created if there exists i' in adjR[j] such that adjL[i] intersects adjL[i'].
        # i.e., they share a common neighbor in Right.
        
        creates_c4 = False
        for i_prime in adjR[j]:
            if adjL[i] & adjL[i_prime]:
                creates_c4 = True
                break
        
        if not creates_c4:
            # Safe to add
            adjL[i].add(j)
            adjR[j].add(i)
            c = random.randint(0, K - 1)
            edges.append([left[i], right[j], c])
            
    return edges
# EVOLVE-BLOCK-END
