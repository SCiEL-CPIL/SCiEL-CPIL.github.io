def construct():
    # Fano plane edges on vertices 0..6
    # Vertices 7 and 8 are isolated
    # This is the unique (up to isomorphism) 3-uniform 3-chromatic hypergraph with 7 edges.
    
    # Define edges as a list of tuples of Python ints
    e1 = (0, 1, 2)
    e2 = (0, 3, 4)
    e3 = (0, 5, 6)
    e4 = (1, 3, 5)
    e5 = (1, 4, 6)
    e6 = (2, 3, 6)
    e7 = (2, 4, 5)
    
    edges = [e1, e2, e3, e4, e5, e6, e7]
    
    # Coloring:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 0
    # 4: 1
    # 5: 1
    # 6: 0
    # 7: 0
    # 8: 0
    # Check non-monochromatic:
    # (0,1,2) -> 0,1,2 OK
    # (0,3,4) -> 0,0,1 OK
    # (0,5,6) -> 0,1,0 OK
    # (1,3,5) -> 1,0,1 OK
    # (1,4,6) -> 1,1,0 OK
    # (2,3,6) -> 2,0,0 OK
    # (2,4,5) -> 2,1,1 OK
    
    colors = [0, 1, 2, 0, 1, 1, 0, 0, 0]
    
    # Ensure pure Python types
    edges = [tuple(int(x) for x in e) for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
