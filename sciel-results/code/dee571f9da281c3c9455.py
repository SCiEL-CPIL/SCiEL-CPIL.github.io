"""Optimized solver for bicolored-forest coloring on the 5x10 toroidal grid."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _validate(coloring, edges):
    """Check adjacency constraint and bicolored forest constraint via union-find per pair."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu < cv:
            pair = (cu, cv)
        else:
            pair = (cv, cu)
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


def _solve_with_dsu(n, edges, adj, k, time_limit=60.0):
    """
    Backtracking solver with incremental DSU per color-pair.
    Key idea: maintain for each color-pair a DSU tracking connectivity.
    When we assign a color to a vertex, we check if it creates a cycle
    in any bicolored subgraph.
    """
    start = time.time()
    coloring = [-1] * n
    # For each pair (c1, c2), maintain a DSU
    # We'll use a dict: (c1, c2) -> {vertex: parent}
    # But that's expensive. Instead, we'll check cycles by BFS in the bicolored subgraph.
    
    # Order vertices: use a fixed good ordering
    # For a grid, process row by row
    order = list(range(n))
    # Sort by a heuristic: prefer vertices with more already-constrained neighbors
    # Actually, let's use a static ordering that works well for grids
    # Process in a snake pattern
    order = []
    for i in range(5):
        row = list(range(i * 10, (i + 1) * 10))
        if i % 2 == 1:
            row.reverse()
        order.extend(row)
    
    # Precompute for each vertex in order, which neighbors come before it in order
    pos_of = [0] * n
    for i, v in enumerate(order):
        pos_of[v] = i
    
    # For fast cycle checking: given a partial coloring and a new assignment (v, c),
    # check if it creates a bicolored cycle.
    # A bicolored cycle is created if, for some neighbor nb with color d != c,
    # there exists a path from v to nb in the subgraph induced by colors {c, d}
    # that doesn't use the edge (v, nb) directly.
    
    def has_path_in_bicolor_subgraph(src, dst, c1, c2, coloring_local, adj_local):
        """Check if there's a path from src to dst using only vertices colored c1 or c2,
        not using the direct edge (src, dst)."""
        if src == dst:
            return False
        # BFS from src, avoiding the direct edge to dst
        visited = {src}
        queue = [src]
        while queue:
            u = queue.pop(0)
            for w in adj_local[u]:
                if w == dst and u == src:
                    continue  # Skip the direct edge we're checking
                if w in visited:
                    continue
                if coloring_local[w] in (c1, c2):
                    if w == dst:
                        return True
                    visited.add(w)
                    queue.append(w)
        return False
    
    def check_and_assign(v, c, coloring_local):
        """Check if assigning color c to vertex v is valid. Returns True if valid."""
        # Check adjacency
        for nb in adj[v]:
            if coloring_local[nb] == c:
                return False
        
        # Check bicolored cycles
        for nb in adj[v]:
            d = coloring_local[nb]
            if d == -1:
                continue
            if d == c:
                continue
            c1, c2 = (min(c, d), max(c, d))
            # Check if there's a path from v to nb in the {c1, c2} subgraph
            # that doesn't use edge (v, nb)
            if has_path_in_bicolor_subgraph(v, nb, c1, c2, coloring_local, adj):
                return False
        return True
    
    def backtrack(idx):
        if time.time() - start > time_limit:
            return None
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        for c in range(k):
            if not check_and_assign(v, c, coloring):
                continue
            coloring[v] = c
            result = backtrack(idx + 1)
            if result is not None:
                return result
            coloring[v] = -1
        return None
    
    return backtrack(0)


def _solve_with_mrv(n, edges, adj, k, time_limit=60.0):
    """
    CSP solver with MRV heuristic and forward checking.
    """
    start = time.time()
    coloring = [-1] * n
    domains = [set(range(k)) for _ in range(n)]
    
    def has_path_bicolor(src, dst, c1, c2, coloring_local):
        """BFS to check path from src to dst in bicolored subgraph, excluding direct edge."""
        if src == dst:
            return False
        visited = {src}
        stack = [src]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if u == src and w == dst:
                    continue
                if w in visited:
                    continue
                if coloring_local[w] in (c1, c2):
                    if w == dst:
                        return True
                    visited.add(w)
                    stack.append(w)
        return False
    
    def is_valid(v, c, coloring_local):
        """Check if color c can be assigned to vertex v."""
        for nb in adj[v]:
            if coloring_local[nb] == c:
                return False
            d = coloring_local[nb]
            if d == -1:
                continue
            if d == c:
                return False
            c1, c2 = (min(c, d), max(c, d))
            if has_path_bicolor(v, nb, c1, c2, coloring_local):
                return False
        return True
    
    def forward_check(v, c, domains_local, coloring_local):
        """Prune domains of unassigned neighbors. Returns False on wipeout."""
        for nb in adj[v]:
            if coloring_local[nb] != -1:
                continue
            # Remove c from nb's domain
            if c in domains_local[nb]:
                domains_local[nb].discard(c)
                if not domains_local[nb]:
                    return False
        return True
    
    def backtrack(assigned_count):
        if time.time() - start > time_limit:
            return False
        if assigned_count == n:
            return True
        
        # MRV: pick unassigned vertex with smallest domain
        min_size = k + 1
        var = -1
        for v in range(n):
            if coloring[v] == -1:
                sz = len(domains[v])
                if sz < min_size:
                    min_size = sz
                    var = v
                    if sz <= 1:
                        break
                if sz == 0:
                    return False
        
        if var == -1:
            return True
        
        # Try colors in domain
        colors = sorted(domains[var])
        for c in colors:
            if not is_valid(var, c, coloring):
                continue
            
            coloring[var] = c
            old_domain = domains[var]
            domains[var] = set()
            
            # Save domains for undo
            saved = {}
            fc_ok = True
            for nb in adj[var]:
                if coloring[nb] != -1:
                    continue
                saved[nb] = domains[nb].copy()
                if c in domains[nb]:
                    domains[nb].discard(c)
                    if not domains[nb]:
                        fc_ok = False
                        break
            
            if fc_ok:
                if backtrack(assigned_count + 1):
                    return True
            
            # Undo
            coloring[var] = -1
            domains[var] = old_domain
            for nb, dom in saved.items():
                domains[nb] = dom
        
        return False
    
    if backtrack(0):
        return coloring[:]
    return None


def _solve_with_pair_constraints(n, edges, adj, k, time_limit=60.0):
    """
    Solver that precomputes which color-pairs can share edges without creating cycles.
    Uses a more aggressive pruning strategy.
    """
    start = time.time()
    coloring = [-1] * n
    
    # Precompute edge list for each vertex
    # For each pair of colors (c1, c2), track which edges connect them
    
    # Use a simple but effective backtracking with good variable ordering
    # Process vertices in an order that maximizes constraint propagation
    
    # For a 5x10 grid, a good order is to process in a way that
    # each new vertex has many already-colored neighbors
    
    # Let's use a different approach: process column by column
    order = []
    for j in range(10):
        for i in range(5):
            order.append(i * 10 + j)
    
    def check_valid(v, c, coloring_local):
        """Check if color c is valid for vertex v."""
        for nb in adj[v]:
            if coloring_local[nb] == c:
                return False
            d = coloring_local[nb]
            if d == -1:
                continue
            if d == c:
                return False
            # Check for bicolored cycle
            c1, c2 = (min(c, d), max(c, d))
            # BFS from v to nb in {c1,c2} subgraph, excluding edge (v,nb)
            visited = {v}
            stack = [v]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if u == v and w == nb:
                        continue
                    if w in visited:
                        continue
                    if coloring_local[w] in (c1, c2):
                        if w == nb:
                            return False
                        visited.add(w)
                        stack.append(w)
        return True
    
    def backtrack(idx):
        if time.time() - start > time_limit:
            return None
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        for c in range(k):
            if not check_valid(v, c, coloring):
                continue
            coloring[v] = c
            result = backtrack(idx + 1)
            if result is not None:
                return result
            coloring[v] = -1
        return None
    
    return backtrack(0)


def _try_direct_patterns(n, k, edges, adj):
    """Try structured patterns that might work for small k."""
    # For a toroidal grid, try patterns based on (row, col)
    # Pattern: color = (a*row + b*col) mod k
    for a in range(k):
        for b in range(k):
            coloring = []
            for i in range(5):
                for j in range(10):
                    coloring.append((a * i + b * j) % k)
            if _validate(coloring, edges):
                return coloring
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    # Try structured patterns first (very fast)
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 60:
            break
        result = _try_direct_patterns(n, k, edges, adj)
        if result is not None and _validate(result, edges):
            return [int(c) for c in result]
    
    # Try different solvers with increasing k
    # Solver 1: MRV + forward checking
    for k in [4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 50:
            break
        remaining = deadline - time.time() - 10
        tl = min(15, max(2, remaining * 0.5))
        result = _solve_with_mrv(n, edges, adj, k, time_limit=tl)
        if result is not None and _validate(result, edges):
            return [int(c) for c in result]
    
    # Solver 2: Column-by-column ordering
    for k in [4, 5, 6, 7, 8]:
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time() - 10
        tl = min(10, max(2, remaining * 0.5))
        result = _solve_with_pair_constraints(n, edges, adj, k, time_limit=tl)
        if result is not None and _validate(result, edges):
            return [int(c) for c in result]
    
    # Solver 3: Simple backtracking with snake ordering
    for k in [5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 15:
            break
        remaining = deadline - time.time() - 5
        tl = min(8, max(1, remaining * 0.5))
        result = _solve_with_dsu(n, edges, adj, k, time_limit=tl)
        if result is not None and _validate(result, edges):
            return [int(c) for c in result]
    
    # Fallback: use a valid coloring with more colors
    # For a 4-regular graph, 5 colors suffice for proper coloring
    coloring = [i % 5 for i in range(n)]
    if not _validate(coloring, edges):
        # Try a different pattern
        coloring = []
        for i in range(5):
            for j in range(10):
                coloring.append((i + 2 * j) % 5)
        if not _validate(coloring, edges):
            # Use vertex index as color (always valid)
            coloring = list(range(n))
    
    return [int(c) for c in coloring]
