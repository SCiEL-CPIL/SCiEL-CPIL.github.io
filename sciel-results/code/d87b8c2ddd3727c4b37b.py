import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Try n from 18 down to 9
    for n in range(18, 8, -1):
        if time.time() > deadline:
            break
            
        try:
            result = solve_n(n, deadline)
            if result is not None:
                return result
        except Exception as e:
            print(f"Error for n={n}: {e}")
            continue
            
    # Fallback to seed n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    """
    CSP-based backtracking with forward checking using vectorized adjacency matrices.
    Each color class has its own n x n boolean adjacency matrix.
    When assigning edge (u,v) color c, check if it creates a monochromatic C6 in color c.
    """
    # Build edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)
    
    # Adjacency matrices for each color (n x n boolean)
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    
    # For each vertex, track which other vertices are connected by each color
    # This helps in C6 detection
    
    # Precompute: for each pair of vertices (a,b), find all vertices c such that
    # (a,c) and (c,b) are edges. This helps detect when adding (a,b) completes a C6.
    # Actually, for C6 detection: adding edge (u,v) of color c creates a C6 if there
    # exists a path of length 5 from u to v using only color-c edges.
    # Equivalently, there exist vertices p1,p2,p3,p4 such that
    # (u,p1),(p1,p2),(p2,p3),(p3,p4),(p4,v) are all color c.
    
    # More efficient: maintain for each color c, the adjacency matrix A_c.
    # When adding edge (u,v) to color c, check if there's a path of length 5 from u to v
    # in A_c (before adding the edge). This is: (A_c^5)[u,v] > 0.
    # But computing matrix powers is expensive. Instead, use BFS/DFS limited to depth 5.
    
    # Alternative: for each color c, maintain A_c as a numpy array.
    # To check if adding (u,v) to color c creates a C6, we need to check if there's
    # a path u -> w1 -> w2 -> w3 -> w4 -> v using only color-c edges (excluding the new edge).
    # This is equivalent to checking if (A_c^5)[u,v] >= 1 where A_c doesn't include (u,v).
    
    # Let's use a different approach: maintain for each color c, for each pair (u,v),
    # the number of paths of length k from u to v for k=1..5.
    # This gets complex. Let's use a simpler but still effective approach.
    
    # For each color c, maintain A_c (n x n boolean).
    # When assigning edge (u,v) to color c, check if there exists a 5-edge path from u to v
    # in the current A_c (without the new edge). We can do this with a limited BFS.
    
    def would_create_c6(color, u, v):
        """Check if adding edge (u,v) to color creates a monochromatic C6."""
        # We need to find a path of length 5 from u to v using only edges of this color
        # (in the current graph, without the new edge (u,v)).
        # Use BFS from u up to depth 5, see if we can reach v.
        # But we need exactly 5 edges, not at most 5.
        
        # Actually, a C6 is a cycle of exactly 6 edges. Adding edge (u,v) to color c
        # creates a C6 if there's a path of exactly 5 edges from u to v using color-c edges
        # (not using the edge (u,v) itself, since it's not added yet).
        
        # Use BFS tracking depth
        from collections import deque
        visited = np.zeros((n, 6), dtype=bool)  # visited[v][d] = reached v at depth d
        queue = deque()
        queue.append((u, 0))
        visited[u][0] = True
        
        while queue:
            node, depth = queue.popleft()
            if depth == 5:
                if node == v:
                    return True
                continue
            # Explore neighbors via color edges
            neighbors = np.where(adj[color][node])[0]
            for nb in neighbors:
                if nb == v and depth + 1 == 5:
                    return True
                if not visited[nb][depth + 1]:
                    visited[nb][depth + 1] = True
                    queue.append((nb, depth + 1))
        
        return False
    
    # Backtracking search
    colors = [-1] * num_edges
    edge_to_uv = edges  # edges[i] = (u,v)
    
    # Order edges for better pruning: try to assign edges that are most constrained first
    # For now, use natural order
    
    # Count of backtracks for time management
    backtrack_count = 0
    max_backtracks = 5000000
    
    def backtrack(idx):
        nonlocal backtrack_count
        if time.time() > deadline - 2.0:
            return False
        if idx == num_edges:
            return True
        
        u, v = edges[idx]
        
        # Try each color
        for c in range(4):
            if would_create_c6(c, u, v):
                continue
            
            # Assign
            colors[idx] = c
            adj[c][u][v] = True
            adj[c][v][u] = True
            
            if backtrack(idx + 1):
                return True
            
            # Unassign
            adj[c][u][v] = False
            adj[c][v][u] = False
            colors[idx] = -1
            backtrack_count += 1
            
            if backtrack_count > max_backtracks:
                return False
        
        return False
    
    if backtrack(0):
        return {'n': n, 'colors': colors}
    
    return None
