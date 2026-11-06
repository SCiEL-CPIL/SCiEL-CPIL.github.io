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
        
        # Pre-compute edge index for quick lookup? 
        # The edges list is in lexicographic order, so index is just the position.
        
        def has_path_len_5(c, u, v):
            # Check for simple path of length 5 from u to v in graph c
            # u - x1 - x2 - x3 - x4 - v
            # Optimization: If u and v are not connected, return False.
            # Quick check: if v is not in adj[c][u]'s connected component? 
            # For speed, just run DFS.
            
            # If u and v are adjacent, path length 1 exists. 
            # We are checking for path length 5 specifically.
            
            def dfs(cur, depth, visited_mask):
                if depth == 0:
                    return cur == v
                if depth == 1:
                    # Check if v is neighbor of cur
                    return (adj[c][cur] >> v) & 1
                
                # Get neighbors of cur in color c that are not visited
                mask = adj[c][cur] & ~visited_mask
                # Additional prune: if depth == 2, we can check if any neighbor of cur is neighbor of v?
                # No, depth 2 means cur - a - v. 
                # If (adj[c][cur] & adj[c][v]) != 0, path exists.
                # But we need to ensure a is not cur or v.
                # Since graph is simple, common neighbor a is distinct from cur and v.
                if depth == 2:
                    return (adj[c][cur] & adj[c][v]) != 0

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
            
            # Symmetry Breaking
            # Edge (0,1) is the first edge in lexicographic order for n>=2
            # We can enforce color 0 for (0,1)
            # Edge (0,2) can be 0 or 1
            # Edge (0,3) can be 0, 1, 2
            
            # Identify which edge this is for symmetry breaking
            # The edges are generated as:
            # (0,1), (0,2), ..., (0,n-1), (1,2), ...
            # So edge index 0 is (0,1)
            # Edge index 1 is (0,2)
            # Edge index 2 is (0,3)
            
            if idx == 0:
                # (0,1) -> Color 0
                if is_valid(0, u, v):
                    colors[idx] = 0
                    adj[0][u] |= (1 << v)
                    adj[0][v] |= (1 << u)
                    if solve(idx + 1):
                        return True
                    colors[idx] = -1
                    adj[0][u] &= ~(1 << v)
                    adj[0][v] &= ~(1 << u)
                return False
            elif idx == 1:
                # (0,2) -> Color 0 or 1
                for c in [0, 1]:
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
            elif idx == 2:
                # (0,3) -> Color 0, 1, 2
                for c in [0, 1, 2]:
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
            else:
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
