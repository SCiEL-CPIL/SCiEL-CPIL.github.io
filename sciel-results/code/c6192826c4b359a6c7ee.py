    def verify_girth_fast(n, adj, g):
        """Verify no cycle shorter than g using non-backtracking walks from every vertex."""
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Non-backtracking BFS from each vertex
        # We use a list of (node, prev, dist)
        # visited is a set of (node, prev)
        
        for start in range(n):
            visited = set()
            queue = deque()
            for nb in adj[start]:
                visited.add((nb, start))
                queue.append((nb, start, 1))
            
            while queue:
                node, prev, dist = queue.popleft()
                if dist >= g:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (nb, node) in visited:
                        # Found a cycle of length dist + 1
                        if dist + 1 < g:
                            return False
                    else:
                        visited.add((nb, node))
                        if dist + 1 < g:
                            queue.append((nb, node, dist + 1))
        return True
