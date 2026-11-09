"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def solve(max_colors, seed=0):
        """Try to find a valid coloring with at most max_colors colors. Returns coloring or None."""
        rng = random.Random(seed)
        
        # Union-Find structure for each color pair
        uf_parents = {}
        
        def uf_find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def uf_union(parent, x, y):
            rx, ry = uf_find(parent, x), uf_find(parent, y)
            if rx == ry:
                return False
            parent[ry] = rx
            return True
        
        def get_uf(c1, c2):
            if c1 == c2:
                return None
            key = (min(c1, c2), max(c1, c2))
            if key not in uf_parents:
                uf_parents[key] = {}
            return uf_parents[key]
        
        def check_and_add_edge(u, v, cu, cv):
            if cu == cv:
                return False, None
            uf = get_uf(cu, cv)
            if uf is None:
                return False, None
            
            # Check if u and v are already in the same component
            if u in uf and v in uf:
                if uf_find(uf, u) == uf_find(uf, v):
                    return False, None
            
            # Perform the union
            if u not in uf:
                uf[u] = u
            if v not in uf:
                uf[v] = v
            
            old_u = uf[u]
            old_v = uf[v]
            
            if not uf_union(uf, u, v):
                uf[u] = old_u
                uf[v] = old_v
                return False, None
            
            return True, (cu, cv, u, v, old_u, old_v)
        
        def undo_edge(info):
            cu, cv, u, v, old_u, old_v = info
            key = (min(cu, cv), max(cu, cv))
            uf = uf_parents.get(key)
            if uf is not None:
                uf[u] = old_u
                uf[v] = old_v
        
        color = [None] * n
        
        def backtrack(assigned_count):
            if assigned_count == n:
                return True
            
            best_vertex = -1
            best_available = None
            best_count = max_colors + 1
            
            for v in range(n):
                if color[v] is not None:
                    continue
                
                available = []
                for c in range(max_colors):
                    neighbor_conflict = False
                    for u in adj[v]:
                        if color[u] is not None and color[u] == c:
                            neighbor_conflict = True
                            break
                    if neighbor_conflict:
                        continue
                    
                    cycle_conflict = False
                    for u in adj[v]:
                        if color[u] is not None:
                            cu = color[u]
                            if cu != c:
                                uf = get_uf(cu, c)
                                if uf is not None:
                                    if u in uf and v in uf:
                                        if uf_find(uf, u) == uf_find(uf, v):
                                            cycle_conflict = True
                                            break
                    if cycle_conflict:
                        continue
                    
                    available.append(c)
                
                if len(available) < best_count:
                    best_count = len(available)
                    best_vertex = v
                    best_available = available
                    if best_count == 0:
                        break
            
            if best_count == 0:
                return False
            
            rng.shuffle(best_available)
            
            for c in best_available:
                color[best_vertex] = c
                
                undo_info = []
                success = True
                for u in adj[best_vertex]:
                    if color[u] is not None:
                        cu = color[u]
                        valid, info = check_and_add_edge(u, best_vertex, cu, c)
                        if not valid:
                            success = False
                            for info2 in reversed(undo_info):
                                undo_edge(info2)
                            break
                        undo_info.append(info)
                
                if success:
                    if backtrack(assigned_count + 1):
                        return True
                
                color[best_vertex] = None
                for info2 in reversed(undo_info):
                    undo_edge(info2)
            
            return False
        
        if backtrack(0):
            return [int(c) for c in color]
        return None
    
    # Try increasing number of colors
    for max_colors in range(2, 50):
        for seed in range(20):
            result = solve(max_colors, seed)
            if result is not None:
                return result
    
    # Fallback
    return list(range(n))

# EVOLVE-BLOCK-END
