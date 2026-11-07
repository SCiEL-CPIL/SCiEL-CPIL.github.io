"""Fast coloring for degree-4 graph with bicolored-forest constraint."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _check_bicolored_forest(coloring, adj):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    n = P['n']
    colors_used = sorted(set(coloring))
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            parent = list(range(n))
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        return False
                    parent[ru] = rv
    return True


def _try_backtrack(k, adj, deadline):
    """Backtracking search for k-coloring with bicolored-forest constraint."""
    n = P['n']
    
    # Use a smart ordering: vertices with higher degree first, but respect structure
    deg = [len(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    pos_in_order = {v: i for i, v in enumerate(order)}
    
    colors = [-1] * n
    # uf[(c1,c2)] = parent array for union-find on that color pair
    uf = {}
    
    def uf_find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def uf_union(parent, x, y):
        px, py = uf_find(parent, x), uf_find(parent, y)
        if px == py:
            return False
        parent[px] = py
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return colors[:]
        
        v = order[idx]
        forbidden = set()
        for u in adj[v]:
            if pos_in_order[u] < idx:
                forbidden.add(colors[pos_in_order[u]])
        
        for c in range(k):
            if c in forbidden:
                continue
            
            # Check bicolored constraint
            valid = True
            other_colors = set()
            for u in adj[v]:
                if pos_in_order[u] < idx:
                    other_colors.add(colors[pos_in_order[u]])
            
            for c2 in other_colors:
                if c2 == c:
                    continue
                key = (min(c, c2), max(c, c2))
                if key not in uf:
                    uf[key] = list(range(n))
                parent = uf[key]
                
                # Find neighbors of v with color c2
                neighbors_c2 = []
                for u in adj[v]:
                    if pos_in_order[u] < idx and colors[pos_in_order[u]] == c2:
                        neighbors_c2.append(u)
                
                # Check if any two such neighbors are connected in the {c,c2} subgraph
                for i in range(len(neighbors_c2)):
                    u1 = neighbors_c2[i]
                    if u1 not in parent:
                        parent[u1] = u1
                    for j in range(i + 1, len(neighbors_c2)):
                        u2 = neighbors_c2[j]
                        if u2 not in parent:
                            parent[u2] = u2
                        if uf_find(parent, u1) == uf_find(parent, u2):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Assign color c to v
            colors[v] = c
            
            # Save state for backtracking
            saved_uf = {}
            for key in uf:
                saved_uf[key] = uf[key][:]
            
            # Update union-find structures
            for u in adj[v]:
                u_pos = pos_in_order[u]
                if u_pos < idx:
                    c2 = colors[u_pos]
                    if c2 == c:
                        continue
                    key = (min(c, c2), max(c, c2))
                    if key not in uf:
                        uf[key] = list(range(n))
                    parent = uf[key]
                    if v not in parent:
                        parent[v] = v
                    if u not in parent:
                        parent[u] = u
                    uf_union(parent, v, u)
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            # Backtrack: restore state
            for key in saved_uf:
                uf[key] = saved_uf[key]
            for key in uf:
                if key not in saved_uf:
                    del uf[key]
            colors[v] = -1
        
        return None
    
    return backtrack(0)


def construct():
    start = time.time()
    deadline = start + 45.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges)
    
    # Fallback: identity coloring (always valid)
    best = list(range(n))
    best_cnt = n
    
    # Try 3 colors first with backtracking
    if time.time() < deadline - 10.0:
        result = _try_backtrack(3, adj, deadline - 5.0)
        if result is not None:
            best = result
            best_cnt = 3
    
    # If 3 colors didn't work, try 4
    if best_cnt > 4 and time.time() < deadline - 10.0:
        result = _try_backtrack(4, adj, deadline - 5.0)
        if result is not None:
            best = result
            best_cnt = 4
    
    # Final fallback: try structured patterns
    if best_cnt > 4:
        for k in [4, 5, 6]:
            if time.time() > deadline:
                break
            for a in range(1, k + 1):
                for b in range(k):
                    if time.time() > deadline:
                        break
                    coloring = [(a * i + b) % k for i in range(n)]
                    ok = all(coloring[u] != coloring[v] for u, v in edges)
                    if not ok:
                        continue
                    if _check_bicolored_forest(coloring, adj):
                        cnt = len(set(coloring))
                        if cnt < best_cnt:
                            best = coloring
                            best_cnt = cnt
                        break
    
    return [int(c) for c in best]

# EVOLVE-BLOCK-END
