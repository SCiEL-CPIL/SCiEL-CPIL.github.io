import time
import random

def construct():
    # Direct construction of two disjoint K_{3,3} graphs.
    # This ensures:
    # - 12 vertices (0..11)
    # - Degree 3 (>= 3)
    # - Triangle-free (bipartite)
    # - Girth 4 (min cycle length 4)
    # - Cycle lengths present: 4 and 6.
    # - Distinct cycle lengths count: 2.
    
    edges = []
    
    # First K_{3,3}: vertices 0,1,2 connected to 3,4,5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3}: vertices 6,7,8 connected to 9,10,11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges
