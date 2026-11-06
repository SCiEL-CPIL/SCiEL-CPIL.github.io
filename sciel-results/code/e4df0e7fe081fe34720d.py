def construct():
    """
    Construct a graph with 40 vertices and 401 edges.
    Strategy: K_{20,20} + one intra-part edge.
    """
    N = 40
    M = 401
    
    # Partition vertices into two sets of 20
    A = list(range(20))
    B = list(range(20, 40))
    
    edges = []
    
    # Add all edges between A and B (400 edges)
    for u in A:
        for v in B:
            edges.append([u, v])
            
    # Add one intra-part edge (1 edge)
    # Any pair in A or B works, book size is 20.
    # Using (0, 1)
    edges.append([0, 1])
    
    # Total edges = 400 + 1 = 401
    return edges
