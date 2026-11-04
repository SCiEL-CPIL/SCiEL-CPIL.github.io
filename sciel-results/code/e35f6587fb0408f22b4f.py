def generate_random_3_regular(n):
    # Start with a cycle
    adj = [set() for _ in range(n)]
    for i in range(n):
        adj[i].add((i + 1) % n)
        adj[(i + 1) % n].add(i)
    
    # We need to add n more edges (since each vertex needs 1 more degree)
    # This is equivalent to finding a perfect matching in the complement of the cycle?
    # Or just randomly add edges.
    
    # Let's just use the configuration model but with retries.
    # The previous implementation was okay, but let's add a retry loop.
    
    for _ in range(100):
        stubs = []
        for i in range(n):
            stubs.extend([i] * 3)
        random.shuffle(stubs)
        edges = set()
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * 3, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return adj, edges
    return None, None
