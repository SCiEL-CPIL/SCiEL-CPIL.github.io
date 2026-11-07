"""SAT-based search for minimum-color valid coloring (bicolored forest constraint)."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
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
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _sat_search(k, adj, n, deadline, var_order=None):
    """DPLL SAT search with forward checking for k-coloring + bicolored forest."""
    if var_order is None:
        var_order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    
    coloring = [-1] * n
    # For each color pair, maintain union-find for cycle detection
    # pair_uf[(a,b)] = list of (u,v) edges in that pair's subgraph
    # We'll check cycles incrementally
    
    # Precompute: for each vertex, its neighbors
    # Use a simpler approach: maintain for each pair a set of edges and check connectivity
    
    # Actually, let's use a cleaner incremental approach:
    # For each pair (a,b), maintain a union-find. When we add vertex v with color c,
    # for each neighbor nb with color cn != c, we add edge (v,nb) to pair (c,cn).
    # If that creates a cycle, it's invalid.
    
    # Union-find per pair, stored as dict: pair_uf[key] = {vertex: parent}
    pair_uf = {}
    
    def uf_find(uf, x):
        if x not in uf:
            uf[x] = x
            return x
        # Path compression
        root = x
        while uf[root] != root:
            root = uf[root]
        while uf[x] != root:
            uf[x], x = root, uf[x]
        return root
    
    def uf_union(uf, x, y):
        rx, ry = uf_find(uf, x), uf_find(uf, y)
        if rx == ry:
            return False
        uf[rx] = ry
        return True
    
    # For undo: we need to save state before each assignment
    # Simpler: just do a full check when needed, but that's slow.
    # Alternative: maintain pair_uf and do careful undo.
    # For now, use a simpler check: after assigning v color c,
    # for each colored neighbor nb with color cn, check if v and nb are
    # already connected in the (c,cn) subgraph (excluding edge v-nb).
    # This is a BFS/DFS check.
    
    def check_cyclic(v, c):
        """Check if assigning color c to v creates a bicolored cycle."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                if cn == c:
                    return False
                continue
            # BFS from v in subgraph of colors {c, cn}, not using edge (v,nb)
            # Check if nb is reachable
            visited = {v}
            stack = [v]
            while stack:
                cur = stack.pop()
                for w in adj[cur]:
                    if w in visited:
                        continue
                    cw = coloring[w]
                    if cw != c and cw != cn:
                        continue
                    # Skip the direct edge v-nb
                    if (cur == v and w == nb) or (cur == nb and w == v):
                        continue
                    if w == nb:
                        return False
                    visited.add(w)
                    stack.append(w)
        return True
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        v = var_order[pos]
        for c in range(k):
            # Check adjacency constraint
            ok = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            coloring[v] = c
            if check_cyclic(v, c):
                if backtrack(pos + 1):
                    return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring[:]
    return None

def _try_pattern_k3(adj, n, deadline):
    """Try structured 3-coloring patterns for the 5x10 cylindrical grid."""
    # Vertex i is at row=i//10, col=i%10
    # Try various period-3 patterns along columns
    
    patterns = []
    # Pattern: color = (row + col) % 3
    patterns.append(lambda r, c: (r + c) % 3)
    # Pattern: color = (2*row + col) % 3
    patterns.append(lambda r, c: (2*r + c) % 3)
    # Pattern: color = (row + 2*col) % 3
    patterns.append(lambda r, c: (r + 2*c) % 3)
    # Pattern: color = (2*row + 2*col) % 3
    patterns.append(lambda r, c: (2*r + 2*c) % 3)
    # Pattern: color = (row + col) % 3 with row offset
    for off in range(3):
        patterns.append(lambda r, c, off=off: (r + off + c) % 3)
    # More patterns
    patterns.append(lambda r, c: (r + c + (r // 2)) % 3)
    patterns.append(lambda r, c: (r + c + (c // 2)) % 3)
    patterns.append(lambda r, c: (2*r + c + r*c) % 3)
    
    for pat in patterns:
        coloring = [pat(i // 10, i % 10) for i in range(n)]
        if _check_coloring(coloring, P['edges']):
            return coloring
    return None

def _local_search_eliminate(coloring, adj, n, deadline, target_k):
    """Try to reduce number of colors to target_k via local search."""
    coloring = coloring[:]
    distinct = set(coloring)
    
    if len(distinct) <= target_k:
        return coloring
    
    # Map colors to 0..target_k-1
    color_map = {c: i for i, c in enumerate(sorted(distinct))}
    if len(distinct) > target_k:
        # Merge colors
        for c in sorted(distinct):
            if c not in color_map:
                continue
            color_map[c] = color_map[c] % target_k
        coloring = [color_map[c] for c in coloring]
    
    # Now try to fix violations by changing colors
    for iteration in range(10000):
        if time.time() > deadline:
            break
        if not _check_coloring(coloring, P['edges']):
            # Find a violation and fix it
            # Try changing a random vertex's color
            import random
            v = random.randint(0, n - 1)
            old_c = coloring[v]
            for new_c in range(target_k):
                if new_c == old_c:
                    continue
                # Check if new_c is valid for v
                ok = True
                for nb in adj[v]:
                    if coloring[nb] == new_c:
                        ok = False
                        break
                if not ok:
                    continue
                coloring[v] = new_c
                if _check_coloring(coloring, P['edges']):
                    break
                coloring[v] = old_c
            else:
                # Try changing a different vertex
                continue
        else:
            # Valid! Try to reduce further
            if len(set(coloring)) <= target_k:
                return coloring
            # Try to merge two colors
            colors_used = sorted(set(coloring))
            if len(colors_used) > target_k:
                # Merge two least-used colors
                from collections import Counter
                cnt = Counter(coloring)
                c1 = min(colors_used, key=lambda c: cnt[c])
                c2 = min((c for c in colors_used if c != c1), key=lambda c: cnt[c])
                # Try merging c2 into c1
                new_coloring = [c1 if c == c2 else c for c in coloring]
                if _check_coloring(new_coloring, P['edges']):
                    coloring = new_coloring
                else:
                    # Try the other direction
                    new_coloring = [c2 if c == c1 else c for c in coloring]
                    if _check_coloring(new_coloring, P['edges']):
                        coloring = new_coloring
                    else:
                        # Skip
                        pass
            else:
                return coloring
    
    return coloring if _check_coloring(coloring, P['edges']) else None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(edges, n)
    
    # First try structured 3-coloring patterns (very fast)
    result = _try_pattern_k3(adj, n, deadline)
    if result is not None:
        if _check_coloring(result, edges):
            return [int(c) for c in result]
    
    # Try SAT search for k=3
    if time.time() < deadline - 5:
        result = _sat_search(3, adj, n, deadline - 5)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Try SAT search for k=4
    if time.time() < deadline - 3:
        result = _sat_search(4, adj, n, deadline - 3)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Fallback: greedy coloring
    coloring = [0] * n
    for v in range(n):
        used = set()
        for nb in adj[v]:
            used.add(coloring[nb])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    
    # Try to reduce
    result = _local_search_eliminate(coloring, adj, n, deadline, 4)
    if result is not None:
        return [int(c) for c in result]
    
    return [int(c) for c in coloring]
