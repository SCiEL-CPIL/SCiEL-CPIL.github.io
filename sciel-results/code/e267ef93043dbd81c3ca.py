def construct():
    edges = []
    # Partition A: 0..6, Partition B: 7..14
    # 1. Complete bipartite K_{7,8}
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    # 2. Edges within A (0..6)
    # Start with all pairs, remove (0,1), (0,2), (3,4), (5,6)
    for u in range(7):
        for v in range(u+1, 7):
            if (u, v) in [(0, 1), (0, 2), (3, 4), (5, 6)]:
                continue
            edges.append([u, v])
            
    # 3. Edges within B (7..14)
    # Complete graph K_8
    for u in range(7, 15):
        for v in range(u+1, 15):
            edges.append([u, v])
            
    return edges
