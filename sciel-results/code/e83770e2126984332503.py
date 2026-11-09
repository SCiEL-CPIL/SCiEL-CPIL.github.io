"""Rewritten program to minimize distinct cycle lengths."""

def construct():
    es = set()
    
    # Create two disjoint K3,3 graphs.
    # K3,3 is 3-regular, bipartite (triangle-free), and has cycle lengths 4 and 6.
    # Disjoint union preserves these properties and degrees.
    # Total distinct cycle lengths: {4, 6} -> count 2.
    
    # Component 1: Vertices 0-5
    # Bipartition: {0, 2, 4} and {1, 3, 5}
    u1 = [0, 2, 4]
    v1 = [1, 3, 5]
    for u in u1:
        for v in v1:
            es.add(tuple(sorted((u, v))))
            
    # Component 2: Vertices 6-11
    # Bipartition: {6, 8, 10} and {7, 9, 11}
    u2 = [6, 8, 10]
    v2 = [7, 9, 11]
    for u in u2:
        for v in v2:
            es.add(tuple(sorted((u, v))))
            
    return [list(e) for e in sorted(es)]
