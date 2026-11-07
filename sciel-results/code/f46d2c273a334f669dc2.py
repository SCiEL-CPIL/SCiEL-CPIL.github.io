"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_cycles(colors, edges):
    """
    Check if the subgraph induced by any pair of colors is a forest.
    """
    n = len(colors)
    color_to_verts = defaultdict(list)
    for i, c in enumerate(colors):
        color_to_verts[c].append(i)
    
    unique_colors = sorted(color_to_verts.keys())
    k = len(unique_colors)
    
    for i in range(k):
        for j in range(i+1, k):
            c1 = unique_colors[i]
            c2 = unique_colors[j]
            
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True

            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not union(u, v):
                        return False
    
    return True

def _try_coloring(max_colors, adj, deadline):
    """
    Try to find a valid coloring using backtracking with max_colors.
    Uses degeneracy ordering for efficiency.
    """
    n = P['n']
    edges = P['edges']
    
    # Calculate degeneracy ordering
    # For a 4-regular graph, degeneracy is 4.
    # We simply sort vertices by degree (all 4) or just use 0..n-1.
    # Actually, let's use a simple heuristic: order by index.
    
    colors = [-1] * n
    
    # Pre-compute neighbors for fast constraint checking
    # neighbors = adj
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == n:
            # Check for bicolored cycles
            if _check_bicolored_cycles(colors, edges):
                return True
            return False
            
        # Heuristic: Try colors that have been used less? Or just 0..max_colors-1
        # Let's try to use existing colors first to minimize distinct colors?
        # No, the task is to minimize distinct colors, so we are already limiting to max_colors.
        # Within max_colors, we just need a valid assignment.
        
        # Optimization: If we have used all max_colors, we must reuse them.
        # If we haven't used all, we can introduce new ones.
        # But since we are trying to find ANY valid coloring with <= max_colors,
        # and we want to minimize the COUNT, the search for max_colors=2,3,4...
        # ensures we find the minimum.
        
        # To speed up, we can try colors 0..max_colors-1.
        for c in range(max_colors):
            # Check if color c is allowed for vertex idx
            valid = True
            for nb in adj[idx]:
                if colors[nb] == c:
                    valid = False
                    break
            if valid:
                colors[idx] = c
                if backtrack(idx + 1):
                    return True
                colors[idx] = -1
        return False

    if backtrack(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin for 90s limit
    
    adj = _build_adj()
    
    # Try 2 colors
    if time.time() < deadline:
        res = _try_coloring(2, adj, deadline)
        if res is not None:
            return res
    
    # Try 3 colors
    if time.time() < deadline:
        res = _try_coloring(3, adj, deadline)
        if res is not None:
            return res
            
    # Try 4 colors
    if time.time() < deadline:
        res = _try_coloring(4, adj, deadline)
        if res is not None:
            return res
            
    # Try 5 colors
    if time.time() < deadline:
        res = _try_coloring(5, adj, deadline)
        if res is not None:
            return res
            
    # Emergency fallback: Identity (50 colors)
    return list(range(P['n']))
# EVOLVE-BLOCK-END
