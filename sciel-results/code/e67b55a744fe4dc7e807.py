def construct():
    # Grötzsch graph edges
    # Vertices 0-4 form a cycle C5
    # Vertices 5-9 are copies of 0-4
    # Vertex 10 is the top vertex
    # Edges:
    # Cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Copies: (0,5), (1,6), (2,7), (3,8), (4,9)
    # Connections: (0,6), (1,7), (2,8), (3,9), (4,5) ? 
    # Standard Mycielski construction:
    # G = (V, E). V' = V U {v' for v in V} U {u}.
    # Edges: E U {(v, w') for (v,w) in E} U {(v', u) for v in V}.
    # So:
    # 1. Cycle edges: (0,1), (1,2), (2,3), (3,4), (4,0)
    # 2. (v, w') edges:
    #    (0,1) -> (0,6), (1,5)
    #    (1,2) -> (1,7), (2,6)
    #    (2,3) -> (2,8), (3,7)
    #    (3,4) -> (3,9), (4,8)
    #    (4,0) -> (4,5), (0,9)
    # 3. (v', u) edges: (5,10), (6,10), (7,10), (8,10), (9,10)
    
    edges_grotzsch = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 6], [1, 5],
        [1, 7], [2, 6],
        [2, 8], [3, 7],
        [3, 9], [4, 8],
        [4, 5], [0, 9],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    
    # Verify triangle-free and chi=4 (assumed correct)
    # Find best disjoint independent sets I, J
    n = 11
    adj = [[False]*n for _ in range(n)]
    for u, v in edges_grotzsch:
        adj[u][v] = True
        adj[v][u] = True
        
    # Get all independent sets
    indep_sets = []
    for mask in range(1 << n):
        is_indep = True
        for u in range(n):
            if mask & (1 << u):
                for v in range(u+1, n):
                    if mask & (1 << v) and adj[u][v]:
                        is_indep = False
                        break
            if not is_indep:
                break
        if is_indep:
            indep_sets.append(mask)
            
    best_score = -1
    best_I = 0
    best_J = 0
    for I in indep_sets:
        for J in indep_sets:
            if I & J == 0:
                score = 2 * bin(I).count('1') + 3 * bin(J).count('1')
                if score > best_score:
                    best_score = score
                    best_I = I
                    best_J = J
                    
    # Construct final graph
    # New vertices 11, 12 (A) and 13, 14, 15 (B)
    # Edges within new: K_{2,3} between A and B
    new_edges = []
    A = [11, 12]
    B = [13, 14, 15]
    for a in A:
        for b in B:
            new_edges.append([a, b])
            
    # Edges between new and old
    for u in range(n):
        if best_I & (1 << u):
            new_edges.append([11, u])
            new_edges.append([12, u])
        if best_J & (1 << u):
            new_edges.append([13, u])
            new_edges.append([14, u])
            new_edges.append([15, u])
            
    all_edges = edges_grotzsch + new_edges
    return all_edges
