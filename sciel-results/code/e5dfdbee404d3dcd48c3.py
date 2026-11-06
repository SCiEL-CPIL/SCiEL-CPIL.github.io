def _search_for_graph(n, time_limit):
    start = time.time()
    deadline = start + time_limit
    max_attempts = 100
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # 1. Initialize
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # 2. Local Search
        result = _local_search(adj, n, deadline)
        if result is not None:
            return result
            
    return None

def _local_search(adj, n, deadline):
    adj = adj[:]
    
    while time.time() < deadline:
        alpha, indep_set = _max_independent_set(adj, n)
        
        if alpha < 10:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return edges
            
        candidates = []
        for i_idx in range(len(indep_set)):
            u = indep_set[i_idx]
            for j_idx in range(i_idx + 1, len(indep_set)):
                v = indep_set[j_idx]
                if (adj[u] & adj[v]) == 0:
                    candidates.append((u, v))
                    
        if candidates:
            u, v = random.choice(candidates)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        else:
            return None

    return None
