"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_bicolored_cycle(coloring, adj):
    """Check if any pair of colors induces a cycle using union-find."""
    n = P['n']
    # Group vertices by color
    color_to_verts = defaultdict(list)
    for v in range(n):
        color_to_verts[coloring[v]].append(v)
    
    # For each pair of colors, check if induced subgraph has a cycle
    colors_used = list(color_to_verts.keys())
    for i in range(len(colors_used)):
        for j in range(i+1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            verts = set(color_to_verts[c1] + color_to_verts[c2])
            # Build subgraph induced by these vertices
            # Use union-find to detect cycles
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False  # cycle detected
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True
            
            # Add edges within the induced subgraph
            for v in verts:
                for w in adj[v]:
                    if w in verts and v < w:
                        if not union(v, w):
                            return True  # cycle found
    return False

def _is_valid(coloring):
    """Check if coloring is valid: adjacent vertices differ, no bicolored cycles."""
    n = P['n']
    adj = _build_adjacency()
    # Check adjacent vertices have different colors
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycles
    if _has_bicolored_cycle(coloring, adj):
        return False
    return True

def _count_distinct(coloring):
    return len(set(coloring))

def _try_coloring_with_k(k, adj, deadline):
    """Try to find a valid coloring using at most k colors (0..k-1)."""
    n = P['n']
    
    # Backtracking search
    coloring = [-1] * n
    
    def backtrack(v):
        if time.time() > deadline:
            return False
        if v == n:
            # Check for bicolored cycles
            if _has_bicolored_cycle(coloring, adj):
                return False
            return True
        
        # Try each color
        for c in range(k):
            # Check if color c is valid for vertex v
            valid = True
            for u in adj[v]:
                if u < v and coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = c
            if backtrack(v + 1):
                return True
            coloring[v] = -1
        
        return False
    
    if backtrack(0):
        return coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    adj = _build_adjacency()
    n = P['n']
    
    # Start with the trivial solution: all different colors
    best_coloring = list(range(n))
    best_k = n
    
    # Try to find colorings with fewer colors
    # Try k from 4 upward (chromatic number lower bound for 4-regular graph is at least 4)
    for k in range(4, best_k):
        if time.time() > deadline - 5:
            break
        coloring = _try_coloring_with_k(k, adj, deadline)
        if coloring is not None:
            if _is_valid(coloring):
                distinct = _count_distinct(coloring)
                if distinct < best_k:
                    best_coloring = coloring
                    best_k = distinct
                break  # Found a valid coloring with k colors, try smaller k next
    
    # Return as list of Python ints
    return [int(c) for c in best_coloring]

# EVOLVE-BLOCK-END
