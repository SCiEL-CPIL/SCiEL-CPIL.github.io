"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_forest(coloring, k, adj_set):
    n = P['n']
    color_classes = defaultdict(list)
    for v in range(n):
        color_classes[coloring[v]].append(v)
    
    for i in range(k):
        for j in range(i + 1, k):
            if i not in color_classes or j not in color_classes:
                continue
            # Union-find on vertices of colors i and j
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            
            for u in color_classes[i]:
                for v in adj_set[u]:
                    if coloring[v] == j:
                        if not union(u, v):
                            return False
    return True

def _validate(coloring, adj_set):
    n = P['n']
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    k = max(coloring) + 1
    return _check_bicolored_forest(coloring, k, adj_set)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    adj = _build_adj()
    adj_set = {v: set(neigh) for v, neigh in adj.items()}
    
    # Try k from 3 upward
    for k in range(3, 16):
        if time.time() > deadline:
            break
        
        # Backtracking with MRV heuristic
        coloring = [-1] * n
        # Precompute domains
        domains = [set(range(k)) for _ in range(n)]
        
        # Order by degree descending for initial assignment
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return _check_bicolored_forest(coloring, k, adj_set)
            
            v = order[idx]
            
            # MRV: pick unassigned vertex with fewest available colors
            best_v = -1
            best_domain = None
            best_size = k + 1
            for i in range(idx, n):
                u = order[i]
                if coloring[u] == -1:
                    # Compute available colors
                    avail = set()
                    for c in range(k):
                        if c not in [coloring[w] for w in adj[u] if coloring[w] != -1]:
                            avail.add(c)
                    if len(avail) < best_size:
                        best_size = len(avail)
                        best_v = u
                        best_domain = avail
                        if best_size <= 1:
                            break
            
            if best_v == -1:
                # All assigned
                return _check_bicolored_forest(coloring, k, adj_set)
            
            if best_size == 0:
                return False
            
            # Swap best_v into position idx
            pos_in_order = order.index(best_v)
            if pos_in_order != idx:
                order[idx], order[pos_in_order] = order[pos_in_order], order[idx]
            
            for c in best_domain:
                coloring[best_v] = c
                
                # Forward check
                feasible = True
                for u in adj[best_v]:
                    if coloring[u] == -1:
                        avail = 0
                        for cc in range(k):
                            if cc not in [coloring[w] for w in adj[u] if coloring[w] != -1]:
                                avail += 1
                        if avail == 0:
                            feasible = False
                            break
                
                if feasible and backtrack(idx + 1):
                    return True
                
                coloring[best_v] = -1
            
            # Restore order
            pos_in_order = order.index(best_v)
            if pos_in_order != idx:
                order[idx], order[pos_in_order] = order[pos_in_order], order[idx]
            
            return False
        
        if backtrack(0):
            if _validate(coloring, adj_set):
                return [int(c) for c in coloring]
    
    # Fallback: greedy 4-coloring
    coloring = [-1] * n
    for v in range(n):
        used = set(coloring[u] for u in adj[v] if coloring[u] != -1)
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    return [int(c) for c in coloring]

# EVOLVE-BLOCK-END
