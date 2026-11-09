"""Construct a 40-vertex, 401-edge graph minimizing the largest book size."""

def construct():
    n = 40
    A = list(range(20))      # vertices 0..19
    B = list(range(20, 40))  # vertices 20..39
    
    edges = set()
    
    # Start with K_{20,20}: all cross edges
    for a in A:
        for b in B:
            edges.add(frozenset([a, b]))
    
    # Remove 10 edges from vertex 0 in A to vertices 20..29 in B
    for i in range(10):
        edges.discard(frozenset([0, 20 + i]))
    
    # Add a star in A: edges (0,1), (0,2), ..., (0,10)
    for i in range(1, 11):
        edges.add(frozenset([0, i]))
    
    # Convert to list of [u, v] pairs
    result = []
    for e in edges:
        u, v = tuple(e)
        result.append([u, v])
    
    return result
