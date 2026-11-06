def construct():
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for y in range(q):
        points.append((1, y, 0))
    points.append((0, 1, 0))
    
    n = len(points)
    adj = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                edges.append([i, j])
                
    # Find non-edge with min common neighbors
    min_common = float('inf')
    best_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 0:
                common = 0
                for k in range(n):
                    if adj[i][k] == 1 and adj[j][k] == 1:
                        common += 1
                if common < min_common:
                    min_common = common
                    best_pair = (i, j)
                    
    if best_pair:
        edges.append([best_pair[0], best_pair[1]])
        
    return edges
