import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # 120s limit, keep 5s margin

    def solve(n):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        m = len(edges)
        colors = [-1] * m
        edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
        
        # adj[c][v] is bitset of neighbors
        adj = [[0] * n for _ in range(4)]
        
        # Symmetry breaking
        fixed = {}
        if n >= 2:
            fixed[(0, 1)] = 0
        if n >= 3:
            fixed[(0, 2)] = 1
        if n >= 4:
            fixed[(0, 3)] = 2
            
        for (i, j), c in fixed.items():
            idx = edge_index[(i, j)]
            colors[idx] = c
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        def has_c6(c, u, v):
            # Check if path of length 4 exists from v to u in color c graph
            # Path: v - a - b - d - e - u ? No, 4 edges means 5 vertices including endpoints?
            # Edges: (v,a), (a,b), (b,d), (d,u). 4 edges.
            # Vertices: v, a, b, d, u.
            # So we need to reach u in 4 steps from v.
            
            visited = 1 << v
            # Use iterative DFS or recursion
            # Recursion depth 4 is safe
            
            def dfs(curr, depth):
                if depth == 4:
                    return bool(adj[c][curr] & (1 << u))
                
                nexts = adj[c][curr] & ~visited
                while nexts:
                    lsb = nexts & -nexts
                    nxt = lsb.bit_length() - 1
                    nexts ^= lsb
                    visited |= lsb
                    if dfs(nxt, depth + 1):
                        return True
                    visited ^= lsb
                return False
                
            return dfs(v, 0)

        def backtrack(k):
            if time.time() - start_time > time_limit:
                return False
                
            if k == m:
                return True
                
            u, v = edges[k]
            
            if colors[k] != -1:
                return backtrack(k + 1)
                
            for c in range(4):
                if has_c6(c, u, v):
                    continue
                    
                colors[k] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(k + 1):
                    return True
                    
                adj[c][u] ^= (1 << v)
                adj[c][v] ^= (1 << u)
                colors[k] = -1
                
            return False
            
        return backtrack(0)

    best_n = 9
    best_colors = None

    # Try larger n first
    for n in [15, 14, 13]:
        if time.time() - start_time > time_limit - 5:
            break
        if solve(n):
            best_n = n
            # Reconstruct colors from the solver? 
            # The solver modifies a local `colors` list. 
            # We need to return it. 
            # Let's modify solve to return the colors list if success.
            pass

    # Refactor solve to return colors
    def solve_with_result(n):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        m = len(edges)
        colors = [-1] * m
        edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
        adj = [[0] * n for _ in range(4)]
        
        fixed = {}
        if n >= 2: fixed[(0, 1)] = 0
        if n >= 3: fixed[(0, 2)] = 1
        if n >= 4: fixed[(0, 3)] = 2
            
        for (i, j), c in fixed.items():
            idx = edge_index[(i, j)]
            colors[idx] = c
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        def has_c6(c, u, v):
            visited = 1 << v
            def dfs(curr, depth):
                if depth == 4:
                    return bool(adj[c][curr] & (1 << u))
                nexts = adj[c][curr] & ~visited
                while nexts:
                    lsb = nexts & -nexts
                    nxt = lsb.bit_length() - 1
                    nexts ^= lsb
                    visited |= lsb
                    if dfs(nxt, depth + 1):
                        return True
                    visited ^= lsb
                return False
            return dfs(v, 0)

        def backtrack(k):
            if time.time() - start_time > time_limit:
                return False
            if k == m:
                return True
            u, v = edges[k]
            if colors[k] != -1:
                return backtrack(k + 1)
            for c in range(4):
                if has_c6(c, u, v):
                    continue
                colors[k] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                if backtrack(k + 1):
                    return True
                adj[c][u] ^= (1 << v)
                adj[c][v] ^= (1 << u)
                colors[k] = -1
            return False

        if backtrack(0):
            return colors
        return None

    for n in [15, 14, 13]:
        if time.time() - start_time > time_limit - 5:
            break
        res = solve_with_result(n)
        if res:
            best_n = n
            best_colors = res
            break

    if best_colors is None:
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
        
    return {'n': best_n, 'colors': best_colors}
