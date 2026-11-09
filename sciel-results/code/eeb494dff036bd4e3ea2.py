"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For a given coloring, check if it's valid:
    # 1. Adjacent vertices have different colors
    # 2. For every pair of colors (a,b), the subgraph induced by vertices with color in {a,b}
    #    (i.e., edges where both endpoints have color a or b) is a forest
    def check_validity(coloring):
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # For each pair of colors, check the induced subgraph is a forest
        colors_used = set(coloring)
        color_pairs = set()
        for c in colors_used:
            for c2 in colors_used:
                if c < c2:
                    color_pairs.add((c, c2))
        
        for a, b in color_pairs:
            # Build subgraph with edges where both endpoints have color in {a, b}
            # Use union-find to detect cycles
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
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == a and cv == b) or (cu == b and cv == a) or (cu == a and cv == a) or (cu == b and cv == b):
                    if not union(u, v):
                        return False
        return True
    
    # Backtracking search for minimum colors
    # Try k = 4, 5, 6, ...
    best_result = list(range(n))  # fallback: all different
    
    for k in range(4, min(n, 12)):
        if time.time() > deadline:
            break
        coloring = [-1] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            # Try colors for vertex idx
            # Order colors: try to reuse existing colors first
            tried = set()
            for c in range(k):
                if c in tried:
                    continue
                # Check if color c is valid for vertex idx
                valid = True
                for nb in adj[idx]:
                    if coloring[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                coloring[idx] = c
                tried.add(c)
                
                if backtrack(idx + 1):
                    return True
                
                coloring[idx] = -1
            
            return False
        
        if backtrack(0):
            if check_validity(coloring):
                best_result = coloring[:]
                break
            else:
                # The backtracking didn't ensure bicolored-cycle constraint
                # Need to add that constraint to backtracking
                pass
    
    # If the simple backtracking didn't work with bicolored constraint,
    # do a more careful search
    if best_result == list(range(n)):
        # Try a constructive approach: use a pattern
        # For a 5x10 grid-like graph, try (row*2 + col) % k patterns
        # Actually, let's try a more systematic backtracking with the full constraint
        
        for k in range(4, min(n, 15)):
            if time.time() > deadline:
                break
            coloring = [-1] * n
            
            # Precompute which color pairs are "active" (have at least one edge)
            def backtrack_full(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                
                for c in range(k):
                    # Check proper coloring
                    valid = True
                    for nb in adj[idx]:
                        if coloring[nb] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    coloring[idx] = c
                    
                    # Check bicolored cycle constraint incrementally
                    # For each color d that is already used, check if adding this edge
                    # creates a cycle in the {c,d} subgraph
                    bicolored_ok = True
                    for nb in adj[idx]:
                        d = coloring[nb]
                        if d == c:
                            continue
                        # Check if edge (idx, nb) with colors (c, d) creates a cycle
                        # in the subgraph induced by colors {c, d}
                        # We need to check if idx and nb are already connected in {c,d} subgraph
                        # This is expensive; for now, do a full check at the end
                        pass
                    
                    if backtrack_full(idx + 1):
                        return True
                    
                    coloring[idx] = -1
                
                return False
            
            if backtrack_full(0):
                if check_validity(coloring):
                    best_result = coloring[:]
                    break
    
    # Final safety: ensure we return a valid result
    if not check_validity(best_result):
        best_result = list(range(n))
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
