def _decompose(edges, n, time_limit=10.0):
    if not edges:
        return []

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    start_time = time.time()
    deadline = start_time + time_limit

    def find_cycle(adj, n):
        """Find a simple cycle using DFS. Returns list of vertices or None."""
        color = {i: 0 for i in range(n)}
        parent = {i: -1 for i in range(n)}
        
        def dfs(u):
            color[u] = 1
            for v in adj[u]:
                if color[v] == 0:
                    parent[v] = u
                    res = dfs(v)
                    if res:
                        return res
                elif color[v] == 1:
                    # Back edge found: u -> v where v is ancestor
                    # Cycle: v -> ... -> u -> v
                    cycle = [v, u]
                    curr = u
                    while curr != v:
                        curr = parent[curr]
                        if curr == -1:
                            break
                        cycle.append(curr)
                    # The list is [v, u, ..., v]
                    # We need to return distinct vertices.
                    # The cycle is the path from v to u in the tree, plus the closing edge.
                    # The vertices are v, ..., u.
                    # My list `cycle` contains v, u, then ancestors of u up to v.
                    # So it is [v, u, ..., v].
                    # We should return [v, ..., u] as the cycle vertices (closing edge implied).
                    # The last element is v, which is the first.
                    # So cycle[:-1] gives [v, ..., u].
                    return cycle[:-1]
            color[u] = 2
            return None

        for i in range(n):
            if color[i] == 0:
                cycle = dfs(i)
                if cycle:
                    return cycle
        return None

    while time.time() < deadline:
        cycle = find_cycle(adj, n)
        if cycle is None:
            break
        
        # Ensure cycle has at least 3 distinct vertices
        if len(set(cycle)) < 3:
             # Should not happen with simple cycle detection, but just in case
             break

        pieces.append([int(v) for v in cycle])

        # Remove edges of the cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            adj[u].discard(v)
            adj[v].discard(u)
            
        # Optional: If the graph becomes disconnected, we still handle it via the main loop
        # which iterates all nodes in find_cycle.

    # Add remaining edges
    remaining_edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                remaining_edges.add((u, v))

    for u, v in sorted(remaining_edges):
        pieces.append([u, v])

    return pieces
