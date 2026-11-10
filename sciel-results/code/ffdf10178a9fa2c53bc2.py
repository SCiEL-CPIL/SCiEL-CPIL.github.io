def has_girth_at_least_10(edges, N):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] > 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] <= 4:
                        queue.append(v)
                else:
                    if v != parent[u]:
                        # Check if this edge creates a short cycle
                        # We only care if dist[u] + dist[v] + 1 < 10
                        # Since dist is from BFS, this is the length of the cycle formed by this edge.
                        if dist[u] + dist[v] + 1 < 10:
                            return False
    return True
