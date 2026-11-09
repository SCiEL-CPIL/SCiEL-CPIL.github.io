import time

def construct():
    n = 16
    # Vertices 0-7 in A, 8-15 in B
    A = list(range(0, 8))
    B = list(range(8, 16))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    return edges
