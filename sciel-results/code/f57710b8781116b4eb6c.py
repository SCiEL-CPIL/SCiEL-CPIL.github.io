import time

P = {'n': 16}

def construct():
    n = P['n']
    # For n=16, the optimal triangle-free graph with diameter 2 is K_{8,8}
    # This has 64 edges.
    # Any graph with more than 64 edges on 16 vertices contains a triangle.
    # Triangles generally lead to non-essential edges in the context of diameter 2 criticality.
    # K_{8,8} is triangle-free, so removing any edge (u,v) increases d(u,v) to 3, 
    # thus increasing the diameter to at least 3.
    
    mid = n // 2
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
            
    return edges
