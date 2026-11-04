# ... (existing imports)

def _check_girth_fast(adj, k, g):
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    start_time = time.time()
    n = 320
    # Try random construction with more attempts
    for attempt in range(500):
        if time.time() - start_time > 110:
            break
        adj = _generate_random_4regular(n)
        if adj is None:
            continue
        if _check_girth_fast(adj, k, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
    
    return _luw_construct(k, g)
