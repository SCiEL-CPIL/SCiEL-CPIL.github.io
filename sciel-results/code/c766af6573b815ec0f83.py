    def verify_girth(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_cycle = float('inf')
        
        # BFS from each vertex
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
                        # Found a cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        # This is valid if u and v are not direct parents/children in the tree
                        # which is checked by parent condition
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return False
            # Optimization: if min_cycle is already >= g, we can stop early?
            # No, a smaller cycle might exist starting from another vertex.
            # But if we found a cycle of length < g, we return False immediately.
            
        return min_cycle >= g
    