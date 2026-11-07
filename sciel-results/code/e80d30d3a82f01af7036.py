def construct():
    N = 15
    # Complete 3-partite graph K_{4,5,6}
    # Part A: vertices 0..3 (size 4)
    # Part B: vertices 4..8 (size 5)
    # Part C: vertices 9..14 (size 6)
    # Edges exist between all pairs of vertices from different parts.
    # Degrees:
    # d(A) = 5 + 6 = 11
    # d(B) = 4 + 6 = 10
    # d(C) = 4 + 5 = 9
    # All degrees are distinct, so d(s) != d(t) for any s, t.
    # Thus, the condition is satisfied for all paths.
    
    edges = []
    # Edges between A and B
    for i in range(4):
        for j in range(4, 9):
            edges.append([i, j])
            
    # Edges between A and C
    for i in range(4):
        for j in range(9, 15):
            edges.append([i, j])
            
    # Edges between B and C
    for i in range(4, 9):
        for j in range(9, 15):
            edges.append([i, j])
            
    return edges
