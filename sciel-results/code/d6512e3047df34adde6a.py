import time

def construct():
    # The best known valid graph for R(C5, K9) > 32 is a union of 8 disjoint K4s.
    # n = 32
    # Edges: all pairs within each block of 4 vertices.
    # This graph is C5-free (components are K4s, no cycles of length 5).
    # Independence number is 8 (one vertex from each K4).
    # Thus alpha = 8 < 9.
    
    n = 32
    edges = []
    for b in range(8):
        offset = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([offset + i, offset + j])
                
    return {'n': n, 'edges': edges}
