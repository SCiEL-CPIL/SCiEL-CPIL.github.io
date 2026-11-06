def construct():
    import time
    import random
    from itertools import combinations
    
    start_time = time.time()
    deadline = 130
    
    n = 16
    adj = [[False]*n for _ in range(n)]
    edges = []
    
    # Base: K7 ∪ K7 ∪ K2 = 43 edges
    for i in range(7):
        for j in range(i+1, 7):
            adj[i][j] = adj[j][i] = True
            edges.append([i, j])
    for i in range(7, 14):
        for j in range(i+1, 14):
            adj[i][j] = adj[j][i] = True
            edges.append([i, j])
    adj[14][15] = adj[15][14] = True
    edges.append([14, 15])
    
    # Candidate cross edges
    candidates = []
    for i in range(7):
        for j in range(7, 14):
            candidates.append((i, j))
    for i in range(7):
        for j in [14, 15]:
            candidates.append((i, j))
    for i in range(7, 14):
        for j in [14, 15]:
            candidates.append((i, j))
    
    random.seed(42)
    random.shuffle(candidates)
    
    for u, v in candidates:
        if time.time() - start_time > deadline:
            break
        if adj[u][v]:
            continue
        
        adj[u][v] = adj[v][u] = True
        # Check for Q3 containing u and v
        found = False
        remaining = [x for x in range(n) if x != u and x != v]
        for combo in combinations(remaining, 6):
            sub = (u, v) + combo
            # Check if induced subgraph on sub is Q3
            edge_count = 0
            degs = [0]*8
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[sub[i]][sub[j]]:
                        edge_count += 1
                        degs[i] += 1
                        degs[j] += 1
            if edge_count == 12 and all(d == 3 for d in degs):
                found = True
                break
        if found:
            adj[u][v] = adj[v][u] = False
        else:
            edges.append([u, v])
    
    return edges
