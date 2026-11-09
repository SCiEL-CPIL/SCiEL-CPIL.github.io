import time

def construct():
    start_time = time.time()
    deadline = 60.0
    
    # Vertices 0..11
    # Two disjoint K_{3,3} graphs
    # K_{3,3} on {0,1,2} x {3,4,5}
    # K_{3,3} on {6,7,8} x {9,10,11}
    
    edges = []
    
    # First K_{3,3}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            if u < v:
                edges.append([u, v])
            else:
                edges.append([v, u])
                
    # Second K_{3,3}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            if u < v:
                edges.append([u, v])
            else:
                edges.append([v, u])
                
    # Sort edges for determinism
    edges.sort()
    
    # Validation checks (quick, to ensure no bugs)
    n = 12
    deg = [0] * n
    adj = {i: set() for i in range(n)}
    
    for e in edges:
        u, v = e[0], e[1]
        if not isinstance(u, int) or not isinstance(v, int):
            return []
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return []
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    # Min degree check
    for i in range(n):
        if deg[i] < 3:
            return []
            
    # Triangle check
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if adj[u].intersection(adj[v]):
                    return []
                    
    # Ensure it's not a tree (must contain a cycle)
    # With 12 vertices and 18 edges, it definitely has cycles.
    # But just to be safe against logic errors in edge generation.
    if len(edges) < n:
        return []

    if time.time() - start_time > deadline:
        return []

    return edges
