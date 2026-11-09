import time

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    best_n = 0
    best_colors = None
    
    for n in [12, 11, 10, 9]:
        if time.time() > deadline - 5:
            break
        colors = _backtrack(n, deadline)
        if colors is not None:
            best_n = n
            best_colors = colors
            break
    
    if best_n == 0:
        n = 9
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}

def _backtrack(n, deadline):
    num_edges = n * (n - 1) // 2
    colors = [-1] * num_edges
    
    # adj[c][u] = bitmask of neighbors of u in color c
    adj = [[0] * n for _ in range(4)]
    
    # Edge list
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def creates_c6(c, u, v):
        """Check if adding (u,v) to color c creates a C6."""
        # Need path u-a-b-d-e-v of length 5 in old graph
        # Use adjacency bitmasks
        adj_c = adj[c]
        
        # For each a neighbor of u
        a_mask = adj_c[u] & ~(1 << u) & ~(1 << v)
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            # For each b neighbor of a, b != u,v,a
            b_mask = adj_c[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            while b_mask:
                b = (b_mask & -b_mask).bit_length() - 1
                b_mask &= b_mask - 1
                # For each d neighbor of b, d != u,v,a,b
                d_mask = adj_c[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                while d_mask:
                    d = (d_mask & -d_mask).bit_length() - 1
                    d_mask &= d_mask - 1
                    # For each e neighbor of d, e != u,v,a,b,d, and e neighbor of v
                    e_mask = adj_c[d] & adj_c[v] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << d)
                    if e_mask:
                        return True
        return False
    
    def add(c, u, v):
        idx = edge_idx(u, v)
        colors[idx] = c
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
    
    def remove(c, u, v):
        idx = edge_idx(u, v)
        colors[idx] = -1
        adj[c][u] &= ~(1 << v)
        adj[c][v] &= ~(1 << u)
    
    def solve(pos):
        if time.time() > deadline:
            return False
        if pos == num_edges:
            return True
        
        u, v = edges[pos]
        for c in range(4):
            if creates_c6(c, u, v):
                continue
            add(c, u, v)
            if solve(pos + 1):
                return True
            remove(c, u, v)
        return False
    
    # Symmetry breaking: fix first 3 edges to break color symmetry
    # Edge 0: (0,1) -> color 0
    # Edge 1: (0,2) -> color 0 or 1 (break color perm symmetry)
    # Edge 2: (0,3) -> color 0, 1, or 2
    
    # Fix (0,1) to color 0
    add(0, 0, 1)
    result = solve(1)
    if not result:
        remove(0, 0, 1)
        return None
    return colors[:]
