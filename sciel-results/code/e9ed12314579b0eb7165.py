"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(edges, n):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_forest(colors, edges, n):
    """Check if for every pair of colors, the induced subgraph is a forest."""
    # Group edges by color pair
    color_pairs = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False  # adjacent same color
        key = (min(cu, cv), max(cu, cv))
        color_pairs[key].append((u, v))
    
    # For each color pair, check if the subgraph is a forest using union-find
    for (c1, c2), edge_list in color_pairs.items():
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
                return False  # cycle detected
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        for u, v in edge_list:
            if not union(u, v):
                return False  # bicolored cycle found
    
    return True

def _count_colors(colors):
    return len(set(colors))

def _validate(colors, edges, n):
    """Full validation: proper coloring + bicolored forest constraint."""
    if len(colors) != n:
        return False
    for c in colors:
        if not isinstance(c, int) or isinstance(c, bool):
            return False
        if c < 0 or c >= 50:
            return False
    # Check proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forest
    return _check_bicolored_forest(colors, edges, n)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges, n)
    
    # Try to find a good coloring
    # Strategy: start with a reasonable initial coloring, then try to reduce colors
    # via local search / recoloring
    
    best_colors = None
    best_num_colors = n + 1
    
    # Initial coloring: try various structured patterns
    # Pattern 1: grid-based (assume 5x10 grid)
    # Let's figure out the grid structure from edges
    # Vertex 0 connects to 1, 7, 43, 49
    # This looks like a torus: 5 rows x 10 cols
    # Row 0: 0-9, Row 1: 10-19, etc? Let's check
    # 0 connects to 1 (right), 7 (down?), 43 (left wrap?), 49 (up wrap?)
    # If 5x10: vertex (r,c) = r*10+c
    # 0=(0,0): neighbors (0,1)=1, (1,0)=10, (4,0)=40, (0,9)=9
    # But 0 connects to 1,7,43,49 - doesn't match simple row-major
    
    # Let me just use a heuristic: try random + local search
    random.seed(42)
    
    # Try multiple initial patterns
    patterns = []
    
    # Pattern: use (i % a) * b + (j % c) style
    # For a 50-vertex graph, try k from 3 upward
    
    for k in range(3, 11):
        # Try a structured coloring: color = (i * 7) % k
        colors = [(i * 7) % k for i in range(n)]
        if _validate(colors, edges, n):
            if k < best_num_colors:
                best_num_colors = k
                best_colors = list(colors)
    
    # Also try random initializations with local search
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        k_target = 3 + attempt // 10
        if k_target > 10:
            break
        
        # Random coloring with k_target colors
        colors = [random.randint(0, k_target - 1) for _ in range(n)]
        
        # Local search: try to fix violations and reduce colors
        for step in range(200):
            if time.time() > deadline:
                break
            
            # Check if valid
            if _validate(colors, edges, n):
                nc = _count_colors(colors)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_colors = list(colors)
                    if nc == 3:
                        return best_colors
                break  # Valid, stop local search for this attempt
            
            # Find a vertex with a conflict and recolor it
            # Check for same-color adjacent
            conflict_found = False
            for u, v in edges:
                if colors[u] == colors[v]:
                    # Try to recolor u
                    old = colors[u]
                    for new_c in range(50):
                        if new_c == old:
                            continue
                        colors[u] = new_c
                        # Quick check: is u's neighborhood ok?
                        ok = True
                        for w in adj[u]:
                            if colors[w] == new_c:
                                ok = False
                                break
                        if ok:
                            conflict_found = True
                            break
                    if not conflict_found:
                        colors[u] = old
                    break
            
            # Also check bicolored cycles - harder to fix locally
            # For now, if we have a proper coloring, try to check and fix
            if _validate(colors, edges, n):
                nc = _count_colors(colors)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_colors = list(colors)
                break
    
    # If no valid coloring found, fall back to trivial valid one
    if best_colors is None:
        best_colors = list(range(n))
    
    # Final validation
    if not _validate(best_colors, edges, n):
        best_colors = list(range(n))
    
    return list(best_colors)
# EVOLVE-BLOCK-END
