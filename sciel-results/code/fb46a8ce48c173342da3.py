"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 50.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute: for each edge, store both endpoints
    # For efficient forest checking, precompute edge indices
    
    def is_forest_fast(verts_set):
        """Check if subgraph induced by verts_set is a forest using union-find."""
        if not verts_set:
            return True
        parent = {v: v for v in verts_set}
        rank = {v: 0 for v in verts_set}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        for u, v in edges:
            if u in verts_set and v in verts_set:
                if not union(u, v):
                    return False
        return True
    
    def check_coloring_fast(colors):
        """Check if a coloring is valid."""
        # Rule 1: adjacent vertices must differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Rule 2: for every pair of colors, induced subgraph is a forest
        color_to_verts = {}
        for i, c in enumerate(colors):
            if c not in color_to_verts:
                color_to_verts[c] = set()
            color_to_verts[c].add(i)
        
        used_colors = sorted(color_to_verts.keys())
        for idx1 in range(len(used_colors)):
            for idx2 in range(idx1, len(used_colors)):
                c1, c2 = used_colors[idx1], used_colors[idx2]
                if c1 == c2:
                    continue  # No edges within same color (adjacent must differ)
                verts_set = color_to_verts[c1] | color_to_verts[c2]
                if not is_forest_fast(verts_set):
                    return False
        return True
    
    # Try small k values with various heuristics
    import random
    random.seed(42)
    
    best_colors = list(range(n))  # fallback
    best_k = n
    
    for k in range(2, 11):
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy 1: Try modular colorings with various offsets and permutations
        for offset in range(k):
            if time.time() > deadline:
                break
            colors_trial = [(i + offset) % k for i in range(n)]
            if check_coloring_fast(colors_trial):
                return [int(c) for c in colors_trial]
        
        # Strategy 2: Try random assignments with local search
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            colors = [random.randint(0, k-1) for _ in range(n)]
            
            # Local search: fix violations
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Check adjacency violations
                for u, v in edges:
                    if colors[u] == colors[v]:
                        # Change one of them
                        new_c = random.randint(0, k-1)
                        while new_c == colors[u] or new_c == colors[v]:
                            new_c = random.randint(0, k-1)
                        # Try changing u
                        old = colors[u]
                        colors[u] = new_c
                        if check_coloring_fast(colors):
                            return [int(c) for c in colors]
                        colors[u] = old
                        # Try changing v
                        old = colors[v]
                        colors[v] = new_c
                        if check_coloring_fast(colors):
                            return [int(c) for c in colors]
                        colors[v] = old
                        improved = True
                        break
                
                if time.time() > deadline:
                    break
                
                # Check bicolored cycles
                color_to_verts = {}
                for i, c in enumerate(colors):
                    if c not in color_to_verts:
                        color_to_verts[c] = set()
                    color_to_verts[c].add(i)
                
                used_colors = sorted(color_to_verts.keys())
                has_cycle = False
                for idx1 in range(len(used_colors)):
                    for idx2 in range(idx1 + 1, len(used_colors)):
                        c1, c2 = used_colors[idx1], used_colors[idx2]
                        verts_set = color_to_verts[c1] | color_to_verts[c2]
                        if not is_forest_fast(verts_set):
                            has_cycle = True
                            break
                    if has_cycle:
                        break
                
                if not has_cycle:
                    # Valid!
                    return [int(c) for c in colors]
        
        # Strategy 3: Backtracking with pruning for small k
        if k <= 5:
            colors = [-1] * n
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            
            # Precompute for each vertex, the neighbors that are already colored
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                
                v = order[idx]
                for c in range(k):
                    # Check adjacency
                    ok = True
                    for u in adj[v]:
                        if colors[u] == c:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
                return False
            
            if backtrack(0):
                if check_coloring_fast(colors):
                    return [int(c) for c in colors]
    
    # Fallback
    return [int(c) for c in best_colors]

# EVOLVE-BLOCK-END
