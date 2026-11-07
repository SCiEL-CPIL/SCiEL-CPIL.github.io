"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute adjacency as sets for faster lookup
    adj_set = {u: set(adj[u]) for u in range(n)}
    
    # Order vertices: try to use a good heuristic
    # Degree-based ordering
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    deadline = time.time() + 85
    
    # Try increasing number of colors starting from 3
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
        result = _backtrack(n, adj, adj_set, order, k, deadline)
        if result is not None:
            # Verify the solution
            if _verify(n, adj, adj_set, result):
                return result
    
    # Fallback: more colors
    for k in range(10, 15):
        if time.time() > deadline - 2:
            break
        result = _backtrack(n, adj, adj_set, order, k, deadline)
        if result is not None:
            if _verify(n, adj, adj_set, result):
                return result
    
    # Last resort
    return [i % 50 for i in range(n)]


def _verify(n, adj, adj_set, color):
    """Verify the coloring is valid."""
    # Check adjacent vertices differ
    for v in range(n):
        for u in adj[v]:
            if u > v and color[u] == color[v]:
                return False
    
    # Check bicolored forest constraint
    colors_used = set(color)
    for i, c1 in enumerate(colors_used):
        for c2 in list(colors_used)[i+1:]:
            if not _is_forest(n, adj, adj_set, color, c1, c2):
                return False
    
    return True


def _is_forest(n, adj, adj_set, color, c1, c2):
    """Check if subgraph induced by vertices with color c1 or c2 is a forest."""
    verts = set()
    for v in range(n):
        if color[v] == c1 or color[v] == c2:
            verts.add(v)
    
    if len(verts) <= 1:
        return True
    
    parent = {v: v for v in verts}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True
    
    for v in verts:
        for u in adj[v]:
            if u in verts and u > v:
                if not union(v, u):
                    return False
    
    return True


def _backtrack(n, adj, adj_set, order, k, deadline):
    """Backtracking search for valid coloring with k colors."""
    color = [-1] * n
    used_colors = set()
    
    def _is_forest_check(c1, c2):
        """Quick check if adding current assignment would violate forest constraint for pair {c1,c2}."""
        verts = set()
        for v in range(n):
            if color[v] == c1 or color[v] == c2:
                verts.add(v)
        
        if len(verts) <= 1:
            return True
        
        parent = {v: v for v in verts}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        
        for v in verts:
            for u in adj[v]:
                if u in verts and u > v:
                    if not union(v, u):
                        return False
        
        return True
    
    def backtrack(idx):
        if time.time() > deadline - 1:
            return None
        
        if idx == n:
            return list(color)
        
        v = order[idx]
        
        # Try colors 0 to k-1
        for c in range(k):
            # Check adjacent constraint
            ok = True
            for u in adj[v]:
                if color[u] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            # Assign
            color[v] = c
            used_colors.add(c)
            
            # Check bicolored forest constraints
            valid = True
            for cu in used_colors:
                if cu == c:
                    continue
                if not _is_forest_check(c, cu):
                    valid = False
                    break
            
            if not valid:
                color[v] = -1
                used_colors.discard(c)
                continue
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            color[v] = -1
            used_colors.discard(c)
        
        return None
    
    return backtrack(0)


# EVOLVE-BLOCK-END
