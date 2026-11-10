def construct():
    # Vertices 0..15
    # A = 0..3, B = 4..15
    edges = []
    
    # Add all edges within A (K4)
    for i in range(4):
        for j in range(i+1, 4):
            edges.append([i, j])
            
    # Add bipartite edges
    # B1 = 4, 5, 6 connected to all A
    for b in [4, 5, 6]:
        for a in range(4):
            edges.append([a, b])
            
    # B2 = 7..15 (9 vertices)
    # Connect each to first 3 vertices in A
    for b in range(7, 16):
        for a in range(3):
            edges.append([a, b])
            
    return edges
