"""Program for the suite task: few colors without any bicolored cycle."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_bicolored_cycles_fast(coloring, edges, n):
    """Fast check: for each color pair, verify induced subgraph is a forest."""
    color_to_verts = defaultdict(set)
    for v in range(n):
        color_to_verts[coloring[v]].add(v)
    
    colors = sorted(color_to_verts.keys())
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            c1, c2 = colors[i], colors[j]
            allowed = color_to_verts[c1] | color_to_verts[c2]
            # Union-Find
            parent = {v: v for v in allowed}
            rank = {v: 0 for v in allowed}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return True  # cycle detected
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1
                return False
            
            for (u, v) in edges:
                if u in allowed and v in allowed:
                    if union(u, v):
                        return True
    return False


def _is_valid(coloring, edges, n):
    """Check proper coloring and no bicolored cycle."""
    for (u, v) in edges:
        if coloring[u] == coloring[v]:
            return False
    if _check_bicolored_cycles_fast(coloring, edges, n):
        return False
    return True


def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency for faster constraint checks
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Strategy: Try 3-colorings systematically
    # The graph is 5x10 grid-like. Try (row + col) % 3 first
    coloring = [0] * n
    for v in range(n):
        row = v // 10
        col = v % 10
        coloring[v] = (row + col) % 3
    
    if _is_valid(coloring, edges, n):
        return [int(c) for c in coloring]
    
    # Try all linear colorings (a*row + b*col) % 3
    for a in range(3):
        for b in range(3):
            if a == 0 and b == 0:
                continue
            cand = [0] * n
            for v in range(n):
                row = v // 10
                col = v % 10
                cand[v] = (a * row + b * col) % 3
            if _is_valid(cand, edges, n):
                return [int(c) for c in cand]
    
    # Try backtracking search with 3 colors
    # Use vertex ordering by degree (highest first) for better pruning
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    color_of = [-1] * n
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = vertex_order[idx]
        # Get available colors (not used by neighbors)
        used_by_neighbors = set()
        for nb in adj[v]:
            if color_of[nb] != -1:
                used_by_neighbors.add(color_of[nb])
        
        for c in range(3):
            if c in used_by_neighbors:
                continue
            
            color_of[v] = c
            
            # Check if this creates a bicolored cycle
            # Only need to check pairs involving color c
            valid = True
            for c2 in range(3):
                if c2 == c:
                    continue
                # Check if adding v with color c creates cycle in (c, c2) subgraph
                # Find all vertices with color c or c2
                allowed = set()
                for u in range(n):
                    if color_of[u] == c or color_of[u] == c2:
                        allowed.add(u)
                
                if len(allowed) < 2:
                    continue
                
                # Union-find on allowed vertices
                parent = {x: x for x in allowed}
                rank = {x: 0 for x in allowed}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra == rb:
                        return True
                    if rank[ra] < rank[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    if rank[ra] == rank[rb]:
                        rank[ra] += 1
                    return False
                
                for (u, w) in edges:
                    if u in allowed and w in allowed:
                        if union(u, w):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            color_of[v] = -1
        
        return False
    
    if backtrack(0):
        return [int(c) for c in color_of]
    
    # Fallback: 4 colors
    coloring = [0] * n
    for v in range(n):
        row = v // 10
        col = v % 10
        coloring[v] = (row + col) % 4
    
    if _is_valid(coloring, edges, n):
        return [int(c) for c in coloring]
    
    # Safety fallback
    return list(range(n))
# EVOLVE-BLOCK-END
