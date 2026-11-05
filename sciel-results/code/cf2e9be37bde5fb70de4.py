import time
import random

def construct():
    n = 18
    # Construct T(18, 3), the complete 3-partite graph with parts of size 6.
    # This graph is K4-free.
    # The largest triangle-free induced subgraph is the union of any two parts,
    # which has size 6 + 6 = 12.
    # Any subset of size 13 must have vertices from all 3 parts,
    # and thus contains a triangle (one from each part).
    
    part_size = 6
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            pi = i // part_size
            pj = j // part_size
            if pi != pj:
                edges.append([i, j])
                
    return edges
