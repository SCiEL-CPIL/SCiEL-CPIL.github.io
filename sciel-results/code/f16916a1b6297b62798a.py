"""Few-color coloring of the degree-4 graph with no bicolored cycles.
Strategy: pattern-based 4-coloring initialization + local search refinement,
with a fast per-pair union-find cycle check."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check(coloring):
    """Validate: adjacent differ and every two-color induced subgraph is a forest."""
    adj = _build_adj()
    n = len(coloring)
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    # Per-pair union-find
    pair_edges = defaultdict(list)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                c1, c2 = coloring[u], coloring[v]
                pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))
    for (a, b), edges in pair_edges.items():
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        vlist = list(verts)
        idx = {v: i for i, v in enumerate(vlist)}
        par = list(range(len(vlist)))
        rn = [0] * len(vlist)
        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        def un(x, y):
            rx, ry = f(x), f(y)
            if rx == ry:
                return False
            if rn[rx] < rn[ry]:
                rx, ry = ry, rx
            par[ry] = rx
            if rn[rx] == rn[ry]:
                rn[rx] += 1
            return True
        for u, v in edges:
            if not un(idx[u], idx[v]):
                return False
    return True

def _fast_check(coloring, adj, n):
    """Fast validation returning True if valid."""
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    pair_edges = defaultdict(list)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                c1, c2 = coloring[u], coloring[v]
                pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))
    for (a, b), edges in pair_edges.items():
        par = {}
        def find(x):
            if x not in par:
                par[x] = x
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            par[ru] = rv
    return True

def _pattern_coloring(adj, n):
    """Generate a 4-coloring based on grid structure.
    The graph is roughly a 7x7 grid with vertex 49 attached.
    Use (row + 2*col) % 4 for grid vertices, assign 49 carefully."""
    # Map vertex to (row, col) for the 7x7 grid (vertices 0-48)
    # From edges: vertex i in row r, col c where i = 7*r + c
    # But the edges show a toroidal-like wrap: 0-7 (row jump), 0-43 (col wrap in 7x7)
    # Let's use a simpler pattern: (i % 7 + (i // 7)) % 4
    coloring = [0] * n
    for i in range(49):
        r, c = divmod(i, 7)
        coloring[i] = (r + 2 * c) % 4
    # Vertex 49: neighbors are 0, 6, 42
    # coloring[0]=(0+0)%4=0, coloring[6]=(0+12)%4=0, coloring[42]=(6+0)%4=2
    # Need color different from 0 and 2: use 1 or 3
    coloring[49] = 1
    return coloring

def _local_search(adj, n, coloring, deadline, max_iter=50000):
    """Local search: try single-vertex color changes to fix bicolored cycles."""
    coloring = coloring[:]
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if _fast_check(coloring, adj, n):
            return coloring
        # Find a bicolored cycle: pick a random edge pair that forms a cycle
        # Simpler: try changing each vertex to a different color and see if it helps
        # Find vertices that are part of a bicolored cycle
        # For efficiency, just try random single-vertex recolors
        import random
        v = random.randint(0, n - 1)
        old_c = coloring[v]
        # Try all other colors
        for new_c in range(4):
            if new_c == old_c:
                continue
            # Check adjacency constraint
            valid_adj = True
            for nb in adj[v]:
                if coloring[nb] == new_c:
                    valid_adj = False
                    break
            if not valid_adj:
                continue
            coloring[v] = new_c
            if _fast_check(coloring, adj, n):
                return coloring
            # Check if this change reduces the number of bicolored cycles
            # (heuristic: just accept if it doesn't create new immediate issues)
            coloring[v] = old_c
        # If no single change works, try a random 2-vertex swap
        v1 = random.randint(0, n - 1)
        v2 = random.randint(0, n - 1)
        if v1 != v2:
            c1, c2 = coloring[v1], coloring[v2]
            if c1 != c2:
                # Check if swapping is valid
                valid = True
                for nb in adj[v1]:
                    if nb != v2 and coloring[nb] == c2:
                        valid = False
                        break
                if valid:
                    for nb in adj[v2]:
                        if nb != v1 and coloring[nb] == c1:
                            valid = False
                            break
                if valid:
                    coloring[v1] = c2
                    coloring[v2] = c1
                    if _fast_check(coloring, adj, n):
                        return coloring
                    coloring[v1] = c1
                    coloring[v2] = c2
    return coloring

def _backtrack(adj, nc, deadline, init_coloring=None):
    """Backtracking search with bicolored-cycle constraint checking."""
    n = P['n']
    if init_coloring is None:
        coloring = [-1] * n
    else:
        coloring = init_coloring[:]

    def get_available_colors(v):
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        return [c for c in range(nc) if c not in used]

    def creates_bicolored_cycle(v, c):
        """Check if assigning color c to vertex v creates a bicolored cycle."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            # BFS in subgraph induced by colors {c, cn}
            # Check if there's a path from v to nb using only vertices colored c or cn
            stack = [v]
            visited = {v}
            found = False
            while stack:
                cur = stack.pop()
                for w in adj[cur]:
                    if cur == v and w == nb:
                        continue
                    if w == nb:
                        found = True
                        break
                    cw = coloring[w]
                    if cw in (c, cn) and w not in visited:
                        visited.add(w)
                        stack.append(w)
                if found:
                    break
            if found:
                return True
        return False

    def dfs():
        if time.time() > deadline:
            return None
        # Select uncolored vertex with fewest available colors
        min_count = nc + 1
        best_v = -1
        best_colors = []
        for v in range(n):
            if coloring[v] == -1:
                avail = get_available_colors(v)
                if len(avail) == 0:
                    return None
                if len(avail) < min_count:
                    min_count = len(avail)
                    best_v = v
                    best_colors = avail
                    if min_count == 1:
                        break
        if best_v == -1:
            return coloring[:]
        v = best_v
        for c in best_colors:
            coloring[v] = c
            if creates_bicolored_cycle(v, c):
                coloring[v] = -1
                continue
            res = dfs()
            if res is not None:
                return res
            coloring[v] = -1
        return None

    return dfs()

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    adj = _build_adj()
    n = P['n']

    # Step 1: Try pattern-based 4-coloring
    pattern = _pattern_coloring(adj, n)
    if _check(pattern):
        return [int(x) for x in pattern]

    # Step 2: Local search from pattern
    import random
    random.seed(42)
    result = _local_search(adj, n, pattern, deadline, max_iter=20000)
    if result is not None and _check(result):
        return [int(x) for x in result]

    # Step 3: Backtracking with 4 colors
    if time.time() < deadline - 10:
        init = [-1] * n
        # Pre-color some vertices to guide search
        for i in range(49):
            r, c = divmod(i, 7)
            init[i] = (r + 2 * c) % 4
        init[49] = 1
        result = _backtrack(adj, 4, deadline, init)
        if result is not None:
            if _check(result):
                return [int(x) for x in result]

    # Step 4: Try different patterns
    for seed_offset in range(10):
        if time.time() > deadline - 5:
            break
        coloring = [0] * n
        for i in range(49):
            r, c = divmod(i, 7)
            coloring[i] = (r + 2 * c + seed_offset) % 4
        coloring[49] = (seed_offset + 1) % 4
        if _check(coloring):
            return [int(x) for x in coloring]
        # Local search from this pattern
        random.seed(seed_offset)
        result = _local_search(adj, n, coloring, deadline, max_iter=5000)
        if result is not None and _check(result):
            return [int(x) for x in result]

    # Step 5: Backtracking with 5 colors as fallback
    if time.time() < deadline - 5:
        init = [-1] * n
        result = _backtrack(adj, 5, deadline, init)
        if result is not None:
            if _check(result):
                return [int(x) for x in result]

    # Fallback: all distinct colors (always valid)
    return [int(i) for i in range(n)]
