    def girth_check_fast(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = g // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            # Found a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            # Note: This might not be the shortest cycle through this edge,
                            # but if dist[u] + dist[v] + 1 < g, we have a cycle < g.
                            # Is it possible that dist[u] + dist[v] + 1 >= g but there is a shorter cycle?
                            # No, BFS finds shortest paths.
                            # Any cycle formed by (u,v) has length at least dist[u] + dist[v] + 1?
                            # Actually, the cycle length is exactly dist[u] + dist[v] + 1 if the paths are disjoint.
                            # They are disjoint in BFS tree.
                            if dist[u] + dist[v] + 1 < g:
                                return False
        return True
