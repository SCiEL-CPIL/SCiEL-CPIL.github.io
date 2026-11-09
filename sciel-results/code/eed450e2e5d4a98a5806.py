"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest(colors, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest using union-find."""
    color_groups = defaultdict(list)
    for i in range(n):
        color_groups[colors[i]].append(i)
    
    colors_used = sorted(color_groups.keys())
    
    for idx_c1 in range(len(colors_used)):
        for idx_c2 in range(idx_c1 + 1, len(colors_used)):
            c1, c2 = colors_used[idx_c1], colors_used[idx_c2]
            vertices = set(color_groups[c1]) | set(color_groups[c2])
            if len(vertices) < 2:
                continue
            
            parent = {v: v for v in vertices}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True
            
            for u in vertices:
                for v in adj[u]:
                    if v in vertices and u < v:
                        if not union(u, v):
                            return False
    return True

def _is_valid_coloring(colors, adj, n):
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    return _check_bicolor_forest(colors, adj, n)

def _count_distinct(colors):
    return len(set(colors))

def _try_reduce_to_k(colors, k, adj, n, deadline):
    """Try to reduce coloring to use at most k distinct colors via local search."""
    current = colors[:]
    
    # Renumber colors to 0..k-1 if possible
    color_map = {}
    new_colors = []
    for c in current:
        if c not in color_map:
            if len(color_map) < k:
                color_map[c] = len(color_map)
            else:
                return None
        new_colors.append(color_map[c])
    current = new_colors
    
    if not _is_valid_coloring(current, adj, n):
        return None
    
    distinct = _count_distinct(current)
    if distinct <= k:
        return current
    
    # Local search: try to reassign vertices to reduce color count
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        for v in range(n):
            old_c = current[v]
            # Try each existing color
            used_colors = sorted(set(current))
            for tc in used_colors:
                if tc == old_c:
                    continue
                # Check proper coloring
                proper = True
                for u in adj[v]:
                    if current[u] == tc:
                        proper = False
                        break
                if not proper:
                    continue
                
                # Tentatively assign
                current[v] = tc
                if _check_bicolor_forest(current, adj, n):
                    new_d = _count_distinct(current)
                    if new_d < distinct:
                        distinct = new_d
                        improved = True
                        if distinct <= k:
                            return current
                        break
                    else:
                        current[v] = old_c
                else:
                    current[v] = old_c
        if not improved:
            break
    
    if _count_distinct(current) <= k:
        return current
    return None

def _backtrack_small_k(k, adj, n, deadline):
    """Backtracking search for a valid k-coloring."""
    colors = [-1] * n
    
    # Order vertices: use degree-based ordering (all degree 4, so use index)
    # Try a smarter order: vertices with most constraints first
    order = list(range(n))
    
    def bt(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        v = order[idx]
        for c in range(k):
            # Check proper coloring with already-colored neighbors
            valid = True
            for u in adj[v]:
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[v] = c
            
            # Quick check: for each color pair involving c, check if cycle forms
            ok = True
            for c2 in range(k):
                if c2 == c:
                    continue
                # Get all vertices with color c or c2
                verts = set()
                for i in range(n):
                    if colors[i] == c or colors[i] == c2:
                        verts.add(i)
                if len(verts) < 3:
                    continue
                
                parent = {x: x for x in verts}
                def fnd(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                cyc = False
                for u in verts:
                    for w in adj[u]:
                        if w in verts and u < w:
                            ru, rw = fnd(u), fnd(w)
                            if ru == rw:
                                cyc = True
                                break
                            parent[ru] = rw
                    if cyc:
                        break
                if cyc:
                    ok = False
                    break
            
            if ok:
                if bt(idx + 1):
                    return True
            colors[v] = -1
        return False
    
    if bt(0):
        return colors[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1
    
    # Strategy 1: Try backtracking for small k (3, 4, 5)
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
        result = _backtrack_small_k(k, adj, n, deadline)
        if result is not None:
            score = _count_distinct(result)
            if score < best_score:
                best_score = score
                best_result = result[:]
            if best_score <= 3:
                break
    
    # Strategy 2: Pattern-based construction + local search reduction
    if best_score > 4:
        # Try various initial patterns
        patterns = []
        
        # Pattern: (row + col) % 4 on 5x10 grid
        def pat1(i):
            row, col = divmod(i, 10)
            return (row + col) % 4
        patterns.append(pat1)
        
        # Pattern: (row * 2 + col) % 4
        def pat2(i):
            row, col = divmod(i, 10)
            return (row * 2 + col) % 4
        patterns.append(pat2)
        
        # Pattern: (row + 2*col) % 4
        def pat3(i):
            row, col = divmod(i, 10)
            return (row + 2 * col) % 4
        patterns.append(pat3)
        
        # Pattern: i % 4
        def pat4(i):
            return i % 4
        patterns.append(pat4)
        
        # Pattern: (row + col) % 5
        def pat5(i):
            row, col = divmod(i, 10)
            return (row + col) % 5
        patterns.append(pat5)
        
        # Pattern: (row * 3 + col) % 5
        def pat6(i):
            row, col = divmod(i, 10)
            return (row * 3 + col) % 5
        patterns.append(pat6)
        
        for pat in patterns:
            if time.time() > deadline:
                break
            
            colors = [pat(i) for i in range(n)]
            
            # Check if valid
            if not _is_valid_coloring(colors, adj, n):
                continue
            
            score = _count_distinct(colors)
            if score < best_score:
                best_score = score
                best_result = colors[:]
            
            # Try to reduce further
            for target_k in range(3, best_score):
                if time.time() > deadline:
                    break
                reduced = _try_reduce_to_k(colors, target_k, adj, n, deadline)
                if reduced is not None:
                    r_score = _count_distinct(reduced)
                    if r_score < best_score:
                        best_score = r_score
                        best_result = reduced[:]
                    break
            
            if best_score <= 3:
                break
    
    # Fallback: greedy coloring
    if best_result is None:
        colors = [0] * n
        for i in range(n):
            used = set()
            for v in adj[i]:
                if v < i:
                    used.add(colors[v])
            for c in range(10):
                if c not in used:
                    colors[i] = c
                    break
            else:
                colors[i] = max(used) + 1 if used else 0
        
        if _is_valid_coloring(colors, adj, n):
            best_result = colors
            best_score = _count_distinct(colors)
        else:
            best_result = list(range(n))
            best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
