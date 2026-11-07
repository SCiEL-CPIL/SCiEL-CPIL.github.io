def construct():
    edges = []
    
    # Construct two disjoint K3,3 components to minimize distinct cycle lengths to {4, 6}
    
    # Component 1: vertices 0-5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([int(u), int(v)])
            
    # Component 2: vertices 6-11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([int(u), int(v)])
            
    return edges
