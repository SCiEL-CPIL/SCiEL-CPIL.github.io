def _generate_simple_regular(k, n, time_limit=5.0):
    start_time = time.time()
    while time.time() - start_time < time_limit:
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        edges = []
        is_simple = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v or v in adj[u]:
                is_simple = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        
        if is_simple:
            return adj, edges
            
        # Repair
        # We'll try to fix defects
        # This is a simplified repair
        # We'll just restart for now to keep code short and robust
        # If it fails too often, the time limit will catch it.
        
    return None, None
