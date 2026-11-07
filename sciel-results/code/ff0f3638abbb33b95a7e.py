def construct():
    n = 15
    import time
    import random
    
    deadline = time.time() + 75
    
    def check_validity(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    A = list(range(7))
    B = list(range(7, 15))
    
    # Start with K_{7,8}
    edges = set()
    for u in A:
        for v in B:
            edges.add((u, v))
    
    # Try to add intra-part edges
    candidates = []
    for i in range(7):
        for j in range(i+1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidates.append((i, j))
    
    random.seed(42)
    random.shuffle(candidates)
    
    for e in candidates:
        if time.time() > deadline - 15:
            break
        if e in edges:
            continue
        test = edges | {e}
        if check_validity(test, n):
            edges = test
    
    result = [[int(u), int(v)] for u, v in edges]
    return result
