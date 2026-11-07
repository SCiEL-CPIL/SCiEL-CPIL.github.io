"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree descending for better pruning
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    # For each pair of colors, track which edges are "active" (both endpoints have those colors)
    # We need to check that for any pair (c1, c2), the induced subgraph is a forest
    
    def check_bicolored_forest(coloring, k):
        """Check if for every pair of colors, the induced subgraph is a forest."""
        # For each pair of colors, build the subgraph and check for cycles using union-find
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                # Union-Find to detect cycles
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
                        return False  # Cycle detected
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                
                for u in range(n):
                    for v in adj[u]:
                        if u < v:  # Avoid double processing
                            cu, cv = coloring[u], coloring[v]
                            # Check if this edge is in the (c1, c2) subgraph
                            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                                if not union(u, v):
                                    return False
        return True
    
    def backtrack(idx, coloring, k, colors_used):
        """Backtracking with forward checking."""
        if idx == len(vertex_order):
            return check_bicolored_forest(coloring, k)
        
        v = vertex_order[idx]
        neighbors = adj[v]
        
        # Try each color
        for c in range(k):
            # Check if c is compatible with already-colored neighbors
            compatible = True
            for nb in neighbors:
                if coloring[nb] != -1 and coloring[nb] == c:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Assign color
            coloring[v] = c
            new_colors_used = colors_used | (1 << c)
            
            # Forward check: for each uncolored neighbor, ensure it still has at least one valid color
            feasible = True
            for nb in neighbors:
                if coloring[nb] == -1:
                    # Check if nb has at least one available color
                    has_valid = False
                    for c2 in range(k):
                        if c2 == c:  # Can't use same color as v
                            continue
                        # Check nb's other colored neighbors
                        nb_ok = True
                        for nb2 in adj[nb]:
                            if nb2 != v and coloring[nb2] != -1 and coloring[nb2] == c2:
                                nb_ok = False
                                break
                        if nb_ok:
                            has_valid = True
                            break
                    if not has_valid:
                        feasible = False
                        break
            
            if feasible:
                if backtrack(idx + 1, coloring, k, new_colors_used):
                    return True
            
            coloring[v] = -1
        
        return False
    
    # Try increasing number of colors
    best_coloring = None
    deadline = time.time() + 55  # Safety margin
    
    for k in range(2, 50):
        if time.time() > deadline:
            break
        
        coloring = [-1] * n
        if backtrack(0, coloring, k, 0):
            best_coloring = coloring
            break
    
    if best_coloring is None:
        # Fallback: use all distinct colors (always valid)
        best_coloring = list(range(n))
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END

