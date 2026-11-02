def _verify_girth(n, edges, g):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
    
    return True
