"""Solver for few-color graph coloring with no bicolored cycles."""
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']

def _build_adj():
    adj = [[] for _ in range(_N)]
    for u, v in _EDGES:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_no_bicolored_cycles(coloring, adj):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    n = _N
    by_color = defaultdict(list)
    for i in range(n):
        by_color[coloring[i]].append(i)
    
    color_list = list(by_color.keys())
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            c1, c2 = color_list[i], color_list[j]
            verts = set(by_color[c1]) | set(by_color[c2])
            if len(verts) < 3:
                continue
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return False
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1
                return True
            
            for v in verts:
                for w in adj[v]:
                    if v < w and w in verts:
                        if not union(v, w):
                            return False
    return True

def _solve_with_backtrack(adj, max_colors, deadline):
    """Backtracking solver with incremental bicolored-cycle detection."""
    n = _N
    coloring = [-1] * n
    
    # Order: try to use a structured ordering that respects the graph's symmetry
    # Use degree ordering with tie-breaking by vertex index
    order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    
    # For each color pair, maintain a union-find structure incrementally
    # Key: (min(c1,c2), max(c1,c2)) -> (parent_dict, rank_dict)
    # But this is complex; instead use targeted BFS check
    
    sys.setrecursionlimit(10000)
    
    def _check_cycle_involving(v, c, coloring):
        """Check if assigning color c to vertex v creates a bicolored cycle."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            # Check if there's a path from nb to v in the {c, cn} subgraph
            # excluding the direct edge (v, nb)
            # BFS from nb using only vertices colored c or cn
            visited = set()
            stack = [nb]
            visited.add(nb)
            found_cycle = False
            while stack:
                cur = stack.pop()
                for nb2 in adj[cur]:
                    if nb2 == v:
                        # We found a path back to v, which means a cycle
                        found_cycle = True
                        break
                    if coloring[nb2] in (c, cn) and nb2 not in visited:
                        visited.add(nb2)
                        stack.append(nb2)
                if found_cycle:
                    break
            if found_cycle:
                return False
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        v = order[idx]
        used_colors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_colors.add(coloring[nb])
        
        for c in range(max_colors):
            if c in used_colors:
                continue
            coloring[v] = c
            if _check_cycle_involving(v, c, coloring):
                if backtrack(idx + 1):
                    return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring[:]
    return None

def _try_structured_coloring(adj, num_colors):
    """Try structured colorings based on the graph's apparent symmetry."""
    n = _N
    # The graph appears to have a regular structure
    # Try coloring based on vertex position patterns
    
    # Pattern 1: i % num_colors
    coloring = [i % num_colors for i in range(n)]
    # Check adjacency
    valid = True
    for u, v in _EDGES:
        if coloring[u] == coloring[v]:
            valid = False
            break
    if valid and _check_no_bicolored_cycles(coloring, adj):
        return coloring
    
    # Pattern 2: (i // 10) * 2 + (i % 10) % 2 if num_colors >= 2
    if num_colors >= 2:
        coloring = [(i // 10) * 2 + (i % 10) % 2 for i in range(n)]
        coloring = [c % num_colors for c in coloring]
        valid = True
        for u, v in _EDGES:
            if coloring[u] == coloring[v]:
                valid = False
                break
        if valid and _check_no_bicolored_cycles(coloring, adj):
            return coloring
    
    # Pattern 3: Based on the apparent grid structure
    # Vertices 0-49 seem to form a 10x5 or similar grid
    # Try: row = i // 10, col = i % 10
    # Color = (row + col) % num_colors
    coloring = [(i // 10 + i % 10) % num_colors for i in range(n)]
    valid = True
    for u, v in _EDGES:
        if coloring[u] == coloring[v]:
            valid = False
            break
    if valid and _check_no_bicolored_cycles(coloring, adj):
        return coloring
    
    # Pattern 4: (row * 2 + col) % num_colors
    coloring = [((i // 10) * 2 + i % 10) % num_colors for i in range(n)]
    valid = True
    for u, v in _EDGES:
        if coloring[u] == coloring[v]:
            valid = False
            break
    if valid and _check_no_bicolored_cycles(coloring, adj):
        return coloring
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    adj = _build_adj()
    
    # First try structured colorings (fast)
    for nc in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 10:
            break
        result = _try_structured_coloring(adj, nc)
        if result is not None:
            # Verify
            if _check_no_bicolored_cycles(result, adj):
                ok = True
                for u, v in _EDGES:
                    if result[u] == result[v]:
                        ok = False
                        break
                if ok:
                    return [int(x) for x in result]
    
    # Then try backtracking
    for nc in [3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 5:
            break
        result = _solve_with_backtrack(adj, nc, deadline - 5)
        if result is not None:
            if _check_no_bicolored_cycles(result, adj):
                ok = True
                for u, v in _EDGES:
                    if result[u] == result[v]:
                        ok = False
                        break
                if ok:
                    return [int(x) for x in result]
    
    # Fallback: use all distinct colors
    return [int(x) for x in range(_N)]
