"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    edge_list = [(u, v) for u, v in edges]
    
    def check_validity(coloring):
        for u, v in edge_list:
            if coloring[u] == coloring[v]:
                return False
        color_pairs = set()
        for u, v in edge_list:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                color_pairs.add((min(c1, c2), max(c1, c2)))
        for c1, c2 in color_pairs:
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in edge_list:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    ru, rv = find(u), find(v)
                    if ru == rv:
                        return False
                    if ru < rv:
                        parent[rv] = ru
                    else:
                        parent[ru] = rv
        return True
    
    def solve_with_k_colors(k, timeout=60.0):
        """Backtracking with MRV and forward checking for k colors."""
        deadline_local = start_time + timeout
        coloring = [-1] * n
        # For each color pair (c1,c2) with c1<c2, track connected components
        # Using a dict: key=(c1,c2), value=dict mapping vertex->component_id
        # Actually, let's use parent arrays indexed by (c1,c2)
        
        # Precompute: for each vertex and each color, which neighbor colors are used
        # This helps with MRV
        
        # State: coloring array + for each color pair, a DSU
        # We'll use a simpler approach: maintain for each (c1,c2) a set of edges
        # and check for cycles incrementally
        
        # Better: for each color pair (c1,c2), maintain a DSU
        # Key: (c1,c2) where c1 < c2
        dsu_data = {}  # (c1,c2) -> (parent_list, rank_list)
        
        def get_dsu(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            if key not in dsu_data:
                dsu_data[key] = (list(range(n)), [0] * n)
            return dsu_data[key]
        
        def dsu_find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def dsu_union(parent, rank, x, y):
            px, py = dsu_find(parent, x), dsu_find(parent, y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Count of times each color is used
        color_count = [0] * k
        
        def backtrack(assigned):
            if time.time() > deadline_local:
                return None
            
            if assigned == n:
                return coloring[:]
            
            # MRV: find unassigned vertex with fewest available colors
            best_v = -1
            best_avail = None
            best_cnt = k + 1
            
            for v in range(n):
                if coloring[v] != -1:
                    continue
                # Available colors: not used by any neighbor
                used_by_neighbors = set()
                for u in adj[v]:
                    if coloring[u] != -1:
                        used_by_neighbors.add(coloring[u])
                avail = [c for c in range(k) if c not in used_by_neighbors]
                if len(avail) < best_cnt:
                    best_cnt = len(avail)
                    best_v = v
                    best_avail = avail
                    if best_cnt <= 1:
                        break
            
            if best_v == -1:
                return coloring[:]
            
            if not best_avail:
                return None
            
            # Try each available color
            for c in best_avail:
                coloring[best_v] = c
                color_count[c] += 1
                
                # Check for bicolored cycles involving color c
                conflict = False
                saved_states = {}
                
                for u in adj[best_v]:
                    d = coloring[u]
                    if d != -1 and d != c:
                        key = (min(c, d), max(c, d))
                        if key not in dsu_data:
                            dsu_data[key] = (list(range(n)), [0] * n)
                        parent, rank = dsu_data[key]
                        # Save state before modification
                        saved_states[key] = (parent[:], rank[:])
                        
                        if not dsu_union(parent, rank, u, best_v):
                            conflict = True
                            break
                
                if not conflict:
                    result = backtrack(assigned + 1)
                    if result is not None:
                        return result
                
                # Undo
                coloring[best_v] = -1
                color_count[c] -= 1
                for key, (sp, sr) in saved_states.items():
                    dsu_data[key] = (sp, sr)
            
            return None
        
        return backtrack(0)
    
    def try_3_colors(timeout=20.0):
        """Try to find a valid 3-coloring."""
        deadline_local = start_time + timeout
        coloring = [-1] * n
        
        # Use a smarter ordering: sort vertices by degree (all degree 4 here)
        # Then use backtracking with forward checking
        
        # For 3 colors, we can use the fact that the graph is 4-regular
        # Let's try a specific pattern based on the graph structure
        
        # The graph appears to be a 50-vertex 4-regular graph
        # Let's try: color based on position mod 3, then fix conflicts
        
        # Actually, let's just do backtracking with MRV
        def backtrack(assigned):
            if time.time() > deadline_local:
                return None
            
            if assigned == n:
                # Final validation
                for u, v in edge_list:
                    if coloring[u] == coloring[v]:
                        return None
                # Check bicolored forests
                for c1 in range(3):
                    for c2 in range(c1 + 1, 3):
                        parent = list(range(n))
                        def find(x):
                            while parent[x] != x:
                                parent[x] = parent[parent[x]]
                                x = parent[x]
                            return x
                        for u, v in edge_list:
                            cu, cv = coloring[u], coloring[v]
                            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                                ru, rv = find(u), find(v)
                                if ru == rv:
                                    return None
                                if ru < rv:
                                    parent[rv] = ru
                                else:
                                    parent[ru] = rv
                return coloring[:]
            
            # MRV
            best_v = -1
            best_avail = None
            best_cnt = 4
            
            for v in range(n):
                if coloring[v] != -1:
                    continue
                used = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                avail = [c for c in range(3) if c not in used]
                if len(avail) < best_cnt:
                    best_cnt = len(avail)
                    best_v = v
                    best_avail = avail
                    if best_cnt <= 1:
                        break
            
            if best_v == -1:
                return coloring[:]
            if not best_avail:
                return None
            
            for c in best_avail:
                coloring[best_v] = c
                
                # Quick check: does this create a bicolored cycle?
                # For each neighbor with a different color, check the pair
                conflict = False
                for u in adj[best_v]:
                    d = coloring[u]
                    if d != -1 and d != c:
                        # Check if adding edge (best_v, u) to color pair (c,d) creates a cycle
                        # We need to check if best_v and u are already connected in the (c,d) subgraph
                        # This requires maintaining DSU state, which is complex in this simple backtracking
                        # For now, skip incremental check and do full validation at the end
                        pass
                
                result = backtrack(assigned + 1)
                if result is not None:
                    return result
                
                coloring[best_v] = -1
            
            return None
        
        return backtrack(0)
    
    def greedy_4_color():
        """Greedy 4-coloring with MRV ordering."""
        coloring = [-1] * n
        remaining = set(range(n))
        
        while remaining:
            # Find vertex with fewest available colors
            best_v = None
            best_avail = None
            best_cnt = 5
            
            for v in remaining:
                used = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                avail = [c for c in range(4) if c not in used]
                if len(avail) < best_cnt:
                    best_cnt = len(avail)
                    best_v = v
                    best_avail = avail
                    if best_cnt <= 1:
                        break
            
            if best_v is None or not best_avail:
                return None
            
            # Choose color that minimizes future conflicts
            best_c = best_avail[0]
            best_score = float('inf')
            
            for c in best_avail:
                score = 0
                for u in adj[best_v]:
                    if coloring[u] != -1:
                        # Count how many unassigned neighbors of u would be constrained
                        unassigned_neighbors = sum(1 for w in adj[u] if w in remaining and w != best_v)
                        score += unassigned_neighbors
                if score < best_score:
                    best_score = score
                    best_c = c
            
            coloring[best_v] = best_c
            remaining.discard(best_v)
        
        # Validate
        if not check_validity(coloring):
            return None
        return coloring
    
    def local_search_improve(coloring, max_colors, timeout=10.0):
        """Try to reduce colors by recoloring."""
        deadline_local = start_time + timeout
        current = coloring[:]
        
        for _ in range(5000):
            if time.time() > deadline_local:
                break
            
            current_colors = set(current)
            if len(current_colors) <= 3:
                break
            
            # Try to eliminate one color
            target_to_eliminate = None
            for c in sorted(current_colors, key=lambda x: -current.count(x)):
                if current.count(c) > 0:
                    target_to_eliminate = c
                    break
            
            if target_to_eliminate is None:
                break
            
            # Find a vertex with target_to_eliminate that can be recolored
            improved = False
            vertices_with_target = [v for v in range(n) if current[v] == target_to_eliminate]
            
            for v in vertices_with_target:
                # Try to recolor v to a different color
                for new_c in range(max_colors):
                    if new_c == target_to_eliminate:
                        continue
                    # Check if new_c is used by any neighbor
                    if any(current[u] == new_c for u in adj[v]):
                        continue
                    
                    # Try the recoloring
                    old_c = current[v]
                    current[v] = new_c
                    
                    if check_validity(current):
                        improved = True
                        break
                    else:
                        current[v] = old_c
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current
    
    best_coloring = None
    best_count = n + 1
    
    # Try 3 colors first
    if time.time() < deadline:
        coloring = try_3_colors(timeout=20.0)
        if coloring is not None:
            best_coloring = coloring
            best_count = 3
    
    # Try 4 colors
    if best_count > 3 and time.time() < deadline:
        coloring = solve_with_k_colors(4, timeout=40.0)
        if coloring is not None and check_validity(coloring):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Greedy fallback
    if best_coloring is None and time.time() < deadline:
        coloring = greedy_4_color()
        if coloring is not None:
            best_coloring = coloring
            best_count = len(set(coloring))
    
    # Local search to improve
    if best_coloring is not None and best_count > 3 and time.time() < deadline:
        improved = local_search_improve(best_coloring, 4, timeout=10.0)
        if check_validity(improved):
            used = len(set(improved))
            if used < best_count:
                best_count = used
                best_coloring = improved[:]
    
    # Final fallback
    if best_coloring is None:
        # Simple greedy with more colors
        for max_c in [5, 6, 7, 8]:
            coloring = [-1] * n
            valid = True
            for v in range(n):
                used = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                avail = [c for c in range(max_c) if c not in used]
                if not avail:
                    valid = False
                    break
                coloring[v] = avail[0]
            if valid and check_validity(coloring):
                best_coloring = coloring
                best_count = len(set(coloring))
                break
    
    if best_coloring is None:
        best_coloring = list(range(n))
        best_count = n
    
    result = [int(c) for c in best_coloring]
    
    # Final validation
    for u, v in edge_list:
        assert result[u] != result[v]
    
    for c1 in range(best_count):
        for c2 in range(c1 + 1, best_count):
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in edge_list:
                cu, cv = result[u], result[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    ru, rv = find(u), find(v)
                    if ru == rv:
                        raise ValueError(f"Bicolored cycle in colors {c1},{c2}")
                    if ru < rv:
                        parent[rv] = ru
                    else:
                        parent[ru] = rv
    
    return result
# EVOLVE-BLOCK-END
