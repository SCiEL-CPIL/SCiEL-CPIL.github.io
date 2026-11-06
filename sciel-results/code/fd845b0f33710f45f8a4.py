import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer

    def has_simple_path_len_5(adj, c, u, v):
        # adj[c] is a list of bitmasks
        # Check if there is a simple path of length 5 from u to v in graph c
        # DFS
        # To speed up, we can use a recursive function with memoization?
        # State: (current_node, depth, visited_mask)
        # visited_mask is an integer bitmask.
        
        # Optimization: If u and v are not connected in 5 steps, return False.
        # We can do a BFS from u up to depth 5 and see if v is reached, 
        # but BFS doesn't track simple paths easily without storing paths.
        # However, for depth 5, the number of paths is small.
        
        # Let's use a stack for DFS to avoid recursion limit issues, 
        # though recursion depth 5 is fine.
        
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            if node == v:
                return False # Path too short
            
            neighbors = adj[c][node] & ~visited
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                if dfs(next_node, depth + 1, visited | lsb):
                    return True
            return False
        
        return dfs(u, 0, 1 << u)

    def solve(n):
        # adj[c][u] is bitmask of neighbors
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        
        # Symmetry breaking
        # Fix colors for the first few edges
        # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # We can fix:
        # c(0,1)=0
        # c(0,2)=0
        # c(0,3)=0
        # c(1,2)=0
        # c(1,3)=1
        # c(2,3)=1
        # This uses colors 0 and 1. Colors 2 and 3 are free for later edges.
        
        fixed_edges = [
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (1, 2, 0),
            (1, 3, 1),
            (2, 3, 1)
        ]
        
        # Apply fixed edges
        for u, v, c in fixed_edges:
            if u >= n or v >= n:
                continue
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[u][v] = c
            colors[v][u] = c
            
        # Determine the next edge to color
        # We iterate through all edges (i, j) with i < j
        # and skip the fixed ones.
        
        # To make it efficient, we can just iterate i from 0 to n-1, j from i+1 to n-1
        # and check if the edge is already colored.
        
        def backtrack(i, j):
            if time.time() > deadline:
                return None
            
            # Find the next uncolored edge
            while i < n:
                while j < n:
                    if colors[i][j] == -1:
                        break
                    j += 1
                if j < n:
                    break
                i += 1
                j = i + 1
                
            if i == n:
                return colors # All edges colored
            
            # Try all 4 colors for (i, j)
            for c in range(4):
                if has_simple_path_len_5(adj, c, i, j):
                    continue
                
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[i][j] = c
                colors[j][i] = c
                
                res = backtrack(i, j + 1)
                if res:
                    return res
                
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[i][j] = -1
                colors[j][i] = -1
                
            return None
        
        return backtrack(0, 1)

    # Try n=17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
        res = solve(n)
        if res:
            flat_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    flat_colors.append(res[i][j])
            return {'n': n, 'colors': flat_colors}
            
    # Fallback: Return a known valid small coloring
    # n=9 is valid in the seed
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
