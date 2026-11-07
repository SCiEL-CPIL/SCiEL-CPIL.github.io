"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, edges, n):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False
        pair = (min(cu, cv), max(cu, cv))
        pair_edges[pair].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        for u, v in elist:
            if not union(u, v):
                return False
    
    return True

def _search_coloring(k, edges, n, deadline):
    """Search for valid coloring with k colors. Returns colors list or None."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    colors = [-1] * n
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return _check_bicolored_forest(colors, edges, n)
        
        v = order[pos]
        for c in range(k):
            valid = True
            for nb in adj[v]:
                if colors[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[v] = c
            if backtrack(pos + 1):
                return True
            colors[v] = -1
        
        return False
    
    if backtrack(0):
        return colors[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    edges = P['edges']
    
    best = list(range(n))
    
    for k in range(3, 50):
        if time.time() > deadline:
            break
        result = _search_coloring(k, edges, n, deadline)
        if result is not None:
            return [int(c) for c in result]
    
    return [int(c) for c in best]
# EVOLVE-BLOCK-END
