def construct():
    # Two disjoint K_{3,3} components
    # Component 1: vertices 0,1,2 (left) and 3,4,5 (right)
    # Component 2: vertices 6,7,8 (left) and 9,10,11 (right)
    edges = []
    
    # First K_{3,3}: left={0,1,2}, right={3,4,5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    
    # Second K_{3,3}: left={6,7,8}, right={9,10,11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
    
    return edges
