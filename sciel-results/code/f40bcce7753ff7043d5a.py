"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def full_check(colors):
        """Full validity check: adjacency + bicolored forest for all color pairs."""
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        color_to_verts = defaultdict(set)
        for i, c in enumerate(colors):
            color_to_verts[c].add(i)
        
        used_colors = sorted(color_to_verts.keys())
        for idx1 in range(len(used_colors)):
            for idx2 in range(idx1, len(used_colors)):
                c1, c2 = used_colors[idx1], used_colors[idx2]
                verts_set = color_to_verts[c1] | color_to_verts[c2]
                if len(verts_set) <= 1:
                    continue
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
    
    # Try backtracking for small k
    def backtrack_solve(k, deadline):
        """Try to find a valid k-coloring using backtracking with constraint propagation."""
        colors = [-1] * n
        color_to_verts = defaultdict(set)
        
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        pos_in_order = {v: i for i, v in enumerate(order)}
        
        def check_pair_forest(c1, c2):
            """Check if subgraph induced by colors c1 and c2 is a forest."""
            verts_set = color_to_verts[c1] | color_to_verts[c2]
            if len(verts_set) <= 1:
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
        
        def is_feasible(v, c):
            """Check if assigning color c to vertex v is feasible."""
            # Adjacency check
            for u in adj[v]:
                if colors[u] == c:
                    return False
            
            # Bicolored forest check for all pairs involving c
            # Only need to check pairs (c, c) and (c, d) for each other color d present
            present_colors = set(color_to_verts.keys())
            
            # Check (c, c): subgraph induced by vertices with color c (plus v)
            # Actually we need to check: after adding v to color c, is the subgraph on color-c vertices a forest?
            # And for each other color d, is the subgraph on (c ∪ d) a forest?
            
            # Efficient: just check all pairs involving c
            for d in present_colors:
                if not check_pair_forest(c, d):
                    return False
            
            return True
        
        def solve(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            # Try colors in order, with some randomization
            color_order = list(range(k))
            if idx % 3 == 0:
                random.shuffle(color_order)
            
            for c in color_order:
                if is_feasible(v, c):
                    colors[v] = c
                    color_to_verts[c].add(v)
                    if solve(idx + 1):
                        return True
                    color_to_verts[c].remove(v)
                    colors[v] = -1
            return False
        
        if solve(0):
            return colors
        return None
    
    # First try small k values with backtracking
    for k in range(3, 8):
        if time.time() > deadline:
            break
        
        # Try multiple random orderings for k=3,4 since backtracking can be expensive
        attempts = 5 if k <= 4 else 2
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            result = backtrack_solve(k, deadline)
            if result is not None:
                if full_check(result):
                    distinct = len(set(result))
                    color_map = {}
                    next_id = 0
                    remapped = []
                    for c in result:
                        if c not in color_map:
                            color_map[c] = next_id
                            next_id += 1
                        remapped.append(color_map[c])
                    return [int(c) for c in remapped]
    
    # Fallback: local search for k=4 or k=5
    for k in [4, 5]:
        if time.time() > deadline:
            break
        
        best_found = None
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with modular coloring
            colors = [i % k for i in range(n)]
            
            # Fix adjacency violations
            for _ in range(200):
                changed = False
                for v in range(n):
                    conflict = False
                    for u in adj[v]:
                        if colors[u] == colors[v]:
                            conflict = True
                            break
                    if conflict:
                        for c in range(k):
                            if c == colors[v]:
                                continue
                            ok = True
                            for u in adj[v]:
                                if colors[u] == c:
                                    ok = False
                                    break
                            if ok:
                                colors[v] = c
                                changed = True
                                break
                if not changed:
                    break
            
            if full_check(colors):
                distinct = len(set(colors))
                if best_found is None or distinct < len(set(best_found)):
                    best_found = colors[:]
            
            # Local search to reduce distinct colors
            for iteration in range(300):
                if time.time() > deadline:
                    break
                
                color_counts = defaultdict(int)
                for c in colors:
                    color_counts[c] += 1
                
                sorted_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])
                improved = False
                
                for target_color in sorted_colors:
                    if time.time() > deadline:
                        break
                    vertices_with_color = [v for v in range(n) if colors[v] == target_color]
                    random.shuffle(vertices_with_color)
                    
                    for v in vertices_with_color:
                        for new_c in range(k):
                            if new_c == target_color:
                                continue
                            test_colors = colors[:]
                            test_colors[v] = new_c
                            if full_check(test_colors):
                                colors[v] = new_c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            if full_check(colors):
                distinct = len(set(colors))
                if best_found is None or distinct < len(set(best_found)):
                    best_found = colors[:]
        
        if best_found is not None:
            distinct = len(set(best_found))
            if full_check(best_found):
                color_map = {}
                next_id = 0
                remapped = []
                for c in best_found:
                    if c not in color_map:
                        color_map[c] = next_id
                        next_id += 1
                    remapped.append(color_map[c])
                return [int(c) for c in remapped]
    
    # Final fallback
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
