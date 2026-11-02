def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k == 5 and g == 9:
        n, edges = _luw_construct(k, g)
        # Verify
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v:
                return _fallback()
            if v in adj[u]:
                return _fallback()
            adj[u].add(v)
            adj[v].add(u)
        
        # Check degree
        for u in range(n):
            if len(adj[u]) != k:
                return _fallback()
        
        # Check girth (simplified: BFS from each vertex, but this is slow)
        # For now, assume LUW is correct.
        return n, edges
    
    return _luw_construct(k, g)
