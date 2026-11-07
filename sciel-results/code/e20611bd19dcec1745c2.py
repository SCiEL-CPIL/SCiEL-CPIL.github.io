"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge set for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def has_bicolored_cycle(colors, c1, c2):
        """Check if subgraph induced by colors c1,c2 has a cycle using union-find."""
        verts = [i for i in range(n) if colors[i] in (c1, c2)]
        if len(verts) < 3:
            return False
        vert_set = set(verts)
        parent = {v: v for v in verts}
        rank = {v: 0 for v in verts}
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
        for u in verts:
            for v in adj[u]:
                if v in vert_set and u < v:
                    if not union(u, v):
                        return True
        return False
    
    def count_violations(colors):
        """Count total violations: bicolored cycles + improper colorings."""
        cnt = 0
        distinct_colors = list(set(colors))
        k = len(distinct_colors)
        for i in range(k):
            for j in range(i+1, k):
                if has_bicolored_cycle(colors, distinct_colors[i], distinct_colors[j]):
                    cnt += 1
        for u, v in edges:
            if colors[u] == colors[v]:
                cnt += 1
        return cnt
    
    def is_valid(colors):
        return count_violations(colors) == 0
    
    import random
    rng = random.Random(42)
    
    # Start with identity coloring (always valid)
    best_colors = list(range(n))
    best_score = n  # number of distinct colors
    
    # Try various initial patterns
    initial_patterns = []
    
    # Modular patterns
    for k in range(2, 15):
        initial_patterns.append([i % k for i in range(n)])
    
    # Random patterns with few colors
    for k in range(2, 8):
        for _ in range(5):
            initial_patterns.append([rng.randint(0, k-1) for _ in range(n)])
    
    # Try to find valid colorings with few colors using backtracking
    def try_backtrack(k_max, timeout=2.0):
        """Try to find a valid k-coloring using backtracking."""
        start_time = time.time()
        colors = [-1] * n
        
        def backtrack(v):
            if time.time() - start_time > timeout:
                return False
            if v == n:
                return is_valid(colors)
            
            # Get available colors
            neighbor_colors = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    neighbor_colors.add(colors[nb])
            
            # Try colors in order, preferring to reuse existing colors
            for c in range(k_max):
                if c in neighbor_colors:
                    continue
                colors[v] = c
                if backtrack(v + 1):
                    return True
            colors[v] = -1
            return False
        
        if backtrack(0):
            return colors[:]
        return None
    
    # First, try backtracking for small k
    for k in range(2, 6):
        if time.time() > deadline:
            break
        result = try_backtrack(k, timeout=3.0)
        if result is not None:
            distinct = len(set(result))
            if distinct < best_score:
                best_score = distinct
                best_colors = result[:]
    
    # Local search from various initial patterns
    for init in initial_patterns:
        if time.time() > deadline:
            break
        
        colors = init[:]
        distinct = len(set(colors))
        
        # Quick validity check
        if is_valid(colors):
            if distinct < best_score:
                best_score = distinct
                best_colors = colors[:]
            continue
        
        # Local search
        current_colors = colors[:]
        no_improve = 0
        max_iters = 3000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a vertex - prioritize vertices involved in violations
            v = rng.randint(0, n-1)
            old_c = current_colors[v]
            
            # Try all colors 0..max_color+1
            max_c = max(current_colors) + 1
            best_new_c = None
            best_new_distinct = len(set(current_colors))
            
            for nc in range(max_c + 1):
                if nc == old_c:
                    continue
                # Quick check: nc must differ from neighbors
                valid_neighbor = True
                for nb in adj[v]:
                    if current_colors[nb] == nc:
                        valid_neighbor = False
                        break
                if not valid_neighbor:
                    continue
                
                trial = current_colors[:]
                trial[v] = nc
                trial_distinct = len(set(trial))
                
                if trial_distinct < best_new_distinct:
                    # Check if valid
                    if is_valid(trial):
                        best_new_c = nc
                        best_new_distinct = trial_distinct
                elif trial_distinct == best_new_distinct and best_new_c is None:
                    if is_valid(trial):
                        best_new_c = nc
            
            if best_new_c is not None:
                current_colors[v] = best_new_c
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 100:
                    break
        
        distinct = len(set(current_colors))
        if is_valid(current_colors):
            if distinct < best_score:
                best_score = distinct
                best_colors = current_colors[:]
    
    # Final polish: try to reduce colors further
    if best_score > 2:
        if time.time() < deadline - 5:
            for _ in range(10):
                if time.time() > deadline:
                    break
                # Try to merge two color classes
                distinct_colors = list(set(best_colors))
                rng.shuffle(distinct_colors)
                for i in range(len(distinct_colors)):
                    for j in range(i+1, len(distinct_colors)):
                        c1, c2 = distinct_colors[i], distinct_colors[j]
                        trial = [c1 if c == c2 else c for c in best_colors]
                        if is_valid(trial):
                            new_distinct = len(set(trial))
                            if new_distinct < best_score:
                                best_score = new_distinct
                                best_colors = trial[:]
    
    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
