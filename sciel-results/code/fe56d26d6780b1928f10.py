    def find_short_cycle(adj, n, max_len):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            
            while queue:
                u = queue.pop(0)
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[v] < dist[u]:
                        # Cycle found
                        # Path from v to u in tree
                        path = []
                        curr = u
                        while curr != v:
                            path.append((curr, parent[curr]))
                            curr = parent[curr]
                        path.append((u, v)) # The non-tree edge
                        # Check length
                        if len(path) < max_len:
                            return path
        return None
