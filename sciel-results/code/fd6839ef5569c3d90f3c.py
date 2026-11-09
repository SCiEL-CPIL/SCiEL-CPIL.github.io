def construct():
    n = 20
    edges = []
    
    # C5 adjacency
    c5_adj = {
        0: {1, 4},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3, 0}
    }
    
    # Blow up each vertex with 4 independent vertices
    # Vertex id: i * 4 + j
    for i in range(5):
        for j in range(4):
            u = i * 4 + j
            for ip in c5_adj[i]:
                if ip > i:
                    for k in range(4):
                        v = ip * 4 + k
                        edges.append([u, v])
    
    return {'n': n, 'edges': edges}
