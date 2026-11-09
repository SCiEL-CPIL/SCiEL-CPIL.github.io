def construct():
    """Construct two disjoint K_{3,3} components on vertices 0..11.
    
    K_{3,3} is bipartite, triangle-free, has minimum degree 3,
    and all simple cycles have length 4 or 6 (exactly 2 distinct lengths).
    """
    edges = []
    # First K_{3,3}: partition {0,1,2} x {3,4,5}
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.append([int(u), int(v)])
    
    # Second K_{3,3}: partition {6,7,8} x {9,10,11}
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.append([int(u), int(v)])
    
    return edges
