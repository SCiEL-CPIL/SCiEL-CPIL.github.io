import time

def construct():
    start_time = time.time()
    
    def solve(n, time_limit):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges
        
        def set_edge(u, v, c, idx):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[idx] = c

        def unset_edge(u, v, c, idx):
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1

        if n >= 4:
            set_edge(0, 1, 0, 0)
            set_edge(0, 2, 1, 1)
            set_edge(0, 3, 2, 2)
            start_idx = 3
        else:
            start_idx = 0

        def has_path_len_5(u, v, c):
            stack = [(u, 0, 1 << u)]
            while stack:
                node, depth, visited = stack.pop()
                if depth == 5:
                    if node == v:
                        return True
                    continue
                neighbors = adj[c][node] & ~visited
                while neighbors:
                    lsb = neighbors & -neighbors
                    nxt = lsb.bit_length() - 1
                    neighbors ^= lsb
                    if depth == 4 and nxt == v:
                        return True
                    stack.append((nxt, depth + 1, visited | lsb))
            return False

        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            if idx == num_edges:
                return True
            u, v = edges[idx]
            for c in range(4):
                if has_path_len_5(u, v, c):
                    continue
                set_edge(u, v, c, idx)
                if backtrack(idx + 1):
                    return True
                unset_edge(u, v, c, idx)
            return False

        if backtrack(start_idx):
            return colors
        return None

    # Try 18 with 60s limit
    colors = solve(18, 60)
    if colors is not None:
        return {'n': 18, 'colors': colors}
        
    # Try 17 with 115s limit
    colors = solve(17, 115)
    if colors is not None:
        return {'n': 17, 'colors': colors}
        
    # Fallback
    return {'n': 1, 'colors': []}
