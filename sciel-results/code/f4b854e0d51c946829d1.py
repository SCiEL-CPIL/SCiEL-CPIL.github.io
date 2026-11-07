"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge list for cycle checking
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def is_bicolored_forest(coloring, c1, c2):
        """Check if subgraph induced by colors c1 and c2 is a forest."""
        # Get vertices with color c1 or c2
        verts = [v for v in range(n) if coloring[v] == c1 or coloring[v] == c2]
        if not verts:
            return True
        # Build subgraph
        sub_adj = defaultdict(set)
        for v in verts:
            for u in adj[v]:
                if u in verts:
                    sub_adj[v].add(u)
        # Check for cycles using DFS
        visited = set()
        def has_cycle(start):
            stack = [(start, -1)]
            while stack:
                node, parent = stack.pop()
                if node in visited:
                    return True
                visited.add(node)
                for nb in sub_adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        return True
                    stack.append((nb, node))
            return False
        for v in verts:
            if v not in visited:
                if has_cycle(v):
                    return False
        return True
    
    def check_all_pairs(coloring, used_colors):
        """Check that all pairs of used colors induce forests."""
        for i in range(len(used_colors)):
            for j in range(i+1, len(used_colors)):
                if not is_bicolored_forest(coloring, used_colors[i], used_colors[j]):
                    return False
        return True
    
    # Try k from 2 upward
    for k in range(2, 15):
        deadline = time.time() + 80  # safety margin
        
        # Use backtracking with MRV
        coloring = [-1] * n
        used_colors = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                # Final check
                uc = sorted(used_colors)
                return check_all_pairs(coloring, uc)
            
            # MRV: find uncolored vertex with fewest options
            best_v = -1
            best_opts = None
            for v in range(idx, n):
                if coloring[v] == -1:
                    opts = [c for c in range(k) if c not in (coloring[u] for u in adj[v] if coloring[u] != -1)]
                    if best_opts is None or len(opts) < len(best_opts):
                        best_v = v
                        best_opts = opts
                        if len(opts) == 0:
                            break
                        if len(opts) == 1:
                            break
            
            if best_v == -1:
                # All remaining are colored
                for v in range(idx, n):
                    if coloring[v] == -1:
                        best_v = v
                        break
                if best_v == -1:
                    uc = sorted(used_colors)
                    return check_all_pairs(coloring, uc)
                best_opts = [c for c in range(k) if c not in (coloring[u] for u in adj[best_v] if coloring[u] != -1)]
            
            if not best_opts:
                return False
            
            # Try each option
            for c in best_opts:
                coloring[best_v] = c
                used_colors.add(c)
                
                # Forward check: verify no bicolored cycle is created
                ok = True
                for c2 in used_colors:
                    if c2 == c:
                        continue
                    if not is_bicolored_forest(coloring, c, c2):
                        ok = False
                        break
                
                if ok:
                    # Recurse
                    if backtrack(idx + 1):
                        return True
                
                coloring[best_v] = -1
                used_colors.discard(c)
            
            return False
        
        if backtrack(0):
            result = [int(coloring[v]) for v in range(n)]
            return result
    
    # Fallback: should not reach here
    return list(range(n))
# EVOLVE-BLOCK-END

