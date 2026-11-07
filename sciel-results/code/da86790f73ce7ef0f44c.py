def construct():
    # Construct two disjoint K3,3 components
    # Component 1: Vertices 0, 1, 2 connected to 3, 4, 5
    # Component 2: Vertices 6, 7, 8 connected to 9, 10, 11
    
    edges = []
    
    # Part 1
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Part 2
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    return edges
