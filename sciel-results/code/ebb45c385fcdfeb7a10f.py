import time

def construct():
    start_time = time.time()
    # Hard deadline: 115 seconds
    hard_deadline = start_time + 115.0

    def attempt(n, time_budget):
        """
        Attempts to find a valid 4-coloring of K_n.
        Returns a list of colors or None.
        """
        local_deadline = time.time() + time_budget
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        colors = [-1] * num_edges
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        
        def has_path_len_4(c, u, v):
            # Check if there is a path of length 4 from u to v in graph c
            # Path: u - x1 - x2 - x3 - v
            # This is equivalent to checking if there is a path of length 3 from u to any neighbor of v (excluding u? no, simple path)
            # Or simply DFS from u with depth 4.
            
            # Optimization: If u and v are not connected by any path of length < 6, we are safe.
            # We specifically check for length 4.
            
            # DFS
            # state: (current_node, depth_remaining, visited_mask)
            # We can prune if depth_remaining is large and graph is sparse.
            
            def dfs(cur, depth, visited_mask):
                if depth == 0:
                    return cur == v
                if depth == 1:
                    # Check if v is neighbor of cur
                    return (adj[c][cur] >> v) & 1
                
                # Get neighbors of cur in color c that are not visited
                mask = adj[c][cur] & ~visited_mask
                # Additional prune: if depth == 2, we can check if any neighbor of cur is neighbor of v?
                # No, depth 2 means cur - a - v. So if (adj[c][cur] & adj[c][v]) != 0, path exists.
                if depth == 2:
                    # Path cur - a - v. We need a common neighbor.
                    # But a cannot be cur or v.
                    # adj[c][cur] contains neighbors.
                    # If (adj[c][cur] & adj[c][v]) != 0, then there is a path of length 2.
                    # But we need to ensure the path is simple.
                    # Since cur != v, and a is neighbor of both, a != cur and a != v is usually true unless multigraph (no).
                    # So if common neighbor exists, path of length 2 exists.
                    return (adj[c][cur] & adj[c][v]) != 0

                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    if dfs(nxt, depth - 1, visited_mask | lsb):
                        return True
                    mask -= lsb
                return False

            return dfs(u, 4, 1 << u)

        def is_valid(c, u, v):
            # Adding (u,v) to color c creates a cycle if there is a path of length 4 from u to v.
            # The new edge (u,v) + path (u...v) of length 4 makes a cycle of length 5?
            # Wait. Edge (u,v) + path of length k forms a cycle of length k+1.
            # We want to avoid C6.
            # So we must avoid creating a cycle of length 6.
            # This happens if there is a path of length 5 from u to v?
            # No.
            # Cycle length = path length + 1.
            # If we add edge (u,v), and there is a path of length L from u to v, we get a cycle of length L+1.
            # We want to avoid L+1 = 6 => L = 5.
            # So we must check for a path of length 5?
            # OR, if there is a path of length 5, we get C6.
            # ALSO, if there is a path of length 1 (direct edge), we get C2 (not a simple cycle in simple graphs, or parallel edge).
            # But the graph is simple.
            # Wait, the task says "No colour class may contain a six-cycle".
            # If I add edge (u,v), and there is ALREADY a path of length 5 between u and v in color c, then the new edge closes a C6.
            # So I must check for a path of length 5.
            
            # Let's re-read carefully.
            # "No colour class may contain a six-cycle"
            # If I assign color c to (u,v), the subgraph G_c gains this edge.
            # If G_c already contained a path of length 5 from u to v, then G_c + (u,v) contains a C6.
            # So `has_path_len_4` in my code above was WRONG. It should be `has_path_len_5`.
            
            pass

        # Correcting the check
        def has_path_len_5(c, u, v):
            # Check for simple path of length 5 from u to v
            # u - x1 - x2 - x3 - x4 - v
            
            # Optimization:
            # If u and v are not connected at all, return False.
            # Use BFS/DFS.
            
            def dfs(cur, depth, visited_mask):
                if depth == 0:
                    return cur == v
                if depth == 1:
                    return (adj[c][cur] >> v) & 1
                
                mask = adj[c][cur] & ~visited_mask
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    if dfs(nxt, depth - 1, visited_mask | lsb):
                        return True
                    mask -= lsb
                return False

            return dfs(u, 5, 1 << u)

        def is_valid(c, u, v):
            return not has_path_len_5(c, u, v)

        def solve(idx):
            if time.time() > local_deadline:
                return False
            if idx == num_edges:
                return True
                
            u, v = edges[idx]
            for c in range(4):
                if is_valid(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if solve(idx + 1):
                        return True
                        
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
            return False

        if solve(0):
            return colors
        return None

    # Try to find the largest N
    # Best known is 17.
    # Try 17 with a generous but bounded time.
    # If it fails, try 16, 15, etc.
    
    # Allocate time:
    # N=17: 60s
    # N=16: 30s
    # N=15: 15s
    # N=12: 5s (fallback)
    
    res = attempt(17, 60)
    if res:
        return {'n': 17, 'colors': res}

    if time.time() > hard_deadline - 40:
        # Not enough time for 16
        pass
    else:
        res = attempt(16, 30)
        if res:
            return {'n': 16, 'colors': res}

    if time.time() > hard_deadline - 15:
        pass
    else:
        res = attempt(15, 15)
        if res:
            return {'n': 15, 'colors': res}

    # Fallback: N=12
    if time.time() > hard_deadline - 5:
        # Just return seed
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    res = attempt(12, 5)
    if res:
        return {'n': 12, 'colors': res}

    # Absolute fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
