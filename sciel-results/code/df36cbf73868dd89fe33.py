"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest_fast(colors, adj, n):
    """Fast check: for every pair of colors, the induced subgraph is a forest."""
    color_groups = defaultdict(list)
    for i in range(n):
        color_groups[colors[i]].append(i)
    
    colors_used = sorted(color_groups.keys())
    
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            v1_set = set(color_groups[c1])
            v2_set = set(color_groups[c2])
            vertices = v1_set | v2_set
            if len(vertices) <= 1:
                continue
            
            # Union-Find with path compression
            parent = {}
            rank = {}
            for v in vertices:
                parent[v] = v
                rank[v] = 0
            
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
            
            # Count edges in induced subgraph; if edges >= vertices, has cycle
            edge_count = 0
            for u in v1_set:
                for v in adj[u]:
                    if v in v2_set or v in v1_set:
                        if u < v:
                            edge_count += 1
            # A forest on k vertices has at most k-1 edges
            if edge_count >= len(vertices):
                return False
            
            # If edge count < vertex count, still need to verify no cycle
            # (could be disconnected with a cycle in one component)
            for u in vertices:
                for v in adj[u]:
                    if v in vertices and u < v:
                        if not union(u, v):
                            return False
    return True

def _is_valid_fast(colors, adj, n):
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    return _check_bicolor_forest_fast(colors, adj, n)

def _count_distinct(colors):
    return len(set(colors))

def _local_search_reduce(colors, adj, n, deadline, target_k):
    """Try to reduce to target_k distinct colors via local search."""
    current = colors[:]
    distinct = _count_distinct(current)
    
    if distinct <= target_k:
        return current, distinct
    
    max_iter = 200
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if distinct <= target_k:
            break
        improved = False
        for i in range(n):
            old_c = current[i]
            # Try assigning to each color in 0..target_k-1
            for c in range(target_k):
                if c == old_c:
                    continue
                # Check proper coloring
                proper = True
                for v in adj[i]:
                    if current[v] == c:
                        proper = False
                        break
                if not proper:
                    continue
                
                current[i] = c
                if _check_bicolor_forest_fast(current, adj, n):
                    new_distinct = _count_distinct(current)
                    if new_distinct < distinct:
                        distinct = new_distinct
                        improved = True
                        break
                    else:
                        current[i] = old_c
                else:
                    current[i] = old_c
            if improved:
                break
        if not improved:
            break
    
    return current, distinct

def _backtrack_color(k, adj, n, deadline, vertex_order=None):
    """Backtracking search for k-coloring with bicolored forest constraint."""
    if vertex_order is None:
        vertex_order = list(range(n))
    
    colors = [-1] * n
    color_counts = [0] * k
    
    def is_feasible(v, c):
        # Check proper coloring
        for u in adj[v]:
            if colors[u] == c:
                return False
        # Check bicolored forest constraint for color c with all existing colors
        for c2 in range(k):
            if c2 == c:
                continue
            if color_counts[c2] == 0:
                continue
            # Check if adding v with color c creates a bicolored cycle with color c2
            # We need to check if the subgraph induced by {v} + vertices of color c2 
            # plus edges among them is still a forest
            # This is expensive; use a heuristic: check if v has >= 2 neighbors in color c2
            # that are connected through color c2 vertices
            neighbors_in_c2 = [u for u in adj[v] if colors[u] == c2]
            if len(neighbors_in_c2) >= 2:
                # Check if these neighbors are in the same component of color c2 subgraph
                # Simple check: if they share a common neighbor in c2, likely cycle
                for u1 in neighbors_in_c2:
                    for u2 in neighbors_in_c2:
                        if u1 < u2:
                            # Check if u1 and u2 are connected in c2-only subgraph
                            # Quick check via BFS limited to c2 vertices
                            visited = set()
                            queue = [u1]
                            visited.add(u1)
                            found = False
                            while queue:
                                cur = queue.pop(0)
                                if cur == u2:
                                    found = True
                                    break
                                for w in adj[cur]:
                                    if w != u1 and w != u2 and colors[w] == c2 and w not in visited:
                                        visited.add(w)
                                        queue.append(w)
                            if found:
                                return False
        return True
    
    def bt(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        v = vertex_order[idx]
        for c in range(k):
            if is_feasible(v, c):
                colors[v] = c
                color_counts[c] += 1
                if bt(idx + 1):
                    return True
                colors[v] = -1
                color_counts[c] -= 1
        return False
    
    if bt(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1
    
    # Try k = 4, 5, 6, ... in order
    for k in range(4, 11):
        if time.time() > deadline:
            break
        
        # Try backtracking with vertex ordering by degree (descending)
        degree_order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        # Try a few different orderings
        for order_seed in range(3):
            if time.time() > deadline:
                break
            
            if order_seed == 0:
                order = degree_order
            elif order_seed == 1:
                order = list(range(n))
            else:
                order = list(reversed(degree_order))
            
            result = _backtrack_color(k, adj, n, deadline, order)
            if result is not None:
                # Verify
                if _is_valid_fast(result, adj, n):
                    distinct = _count_distinct(result)
                    if distinct < best_score:
                        best_score = distinct
                        best_result = result[:]
                    break
        
        if best_result is not None and best_score <= k:
            # Try to polish
            polished, pol_score = _local_search_reduce(best_result, adj, n, deadline, k - 1)
            if pol_score < best_score:
                best_score = pol_score
                best_result = polished[:]
            break
    
    # If backtracking didn't work well, try random restarts with local search
    if best_score > 6:
        for seed in range(500):
            if time.time() > deadline:
                break
            random.seed(seed)
            # Start with random k-coloring
            for k_try in [5, 6, 7]:
                colors = [random.randint(0, k_try - 1) for _ in range(n)]
                if _is_valid_fast(colors, adj, n):
                    result, score = _local_search_reduce(colors, adj, n, deadline, k_try)
                    if score < best_score:
                        best_score = score
                        best_result = result[:]
    
    # Fallback
    if best_result is None:
        best_result = list(range(n))
        best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
