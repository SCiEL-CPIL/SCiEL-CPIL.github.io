import time
from collections import defaultdict, deque

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


def _direct_4coloring():
    """Direct construction: 4-coloring of the 5x10 toroidal grid.
    
    Vertex i is at (row=i//10, col=i%10).
    Edges: horizontal (col+1)%10, vertical (row+1)%5, diagonal (row+1, col+1).
    
    Try coloring: color = (row + 2*col) % 4
    This gives 4 colors. Verify no bicolored cycles.
    """
    n = P['n']
    coloring = [0] * n
    for v in range(n):
        row = v // 10
        col = v % 10
        coloring[v] = (row + 2 * col) % 4
    return coloring


def _try_colorings(nc, deadline, adj):
    """Try to find a valid nc-coloring using backtracking with MRV."""
    n = P['n']
    coloring = [-1] * n
    
    # Precompute: for each vertex, its neighbors
    # MRV heuristic: pick vertex with fewest available colors
    
    def get_available(v):
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        return [c for c in range(nc) if c not in used]
    
    def has_bicolored_cycle(v, c):
        """Check if assigning color c to v creates a bicolored cycle."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            # BFS from v in {c, cn} subgraph, avoiding direct edge (v,nb)
            visited = {v}
            queue = deque([v])
            found = False
            while queue:
                cur = queue.popleft()
                for w in adj[cur]:
                    if cur == v and w == nb:
                        continue
                    if w == nb:
                        found = True
                        break
                    cw = coloring[w]
                    if cw in (c, cn) and w not in visited:
                        visited.add(w)
                        queue.append(w)
                if found:
                    break
            if found:
                return True
        return False
    
    def dfs():
        if time.time() > deadline:
            return None
        
        # Find uncolored vertex with fewest available colors
        min_count = nc + 1
        best_v = -1
        best_colors = []
        
        for v in range(n):
            if coloring[v] == -1:
                avail = get_available(v)
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
            if not has_bicolored_cycle(v, c):
                res = dfs()
                if res is not None:
                    return res
            coloring[v] = -1
        return None

    return dfs()


def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # First: try the direct 4-coloring construction
    c4 = _direct_4coloring()
    if _check(c4):
        return [int(x) for x in c4]
    
    # Try backtracking for 3 colors (likely impossible due to triangles)
    if time.time() < deadline - 20:
        result = _try_colorings(3, deadline - 10, adj)
        if result is not None and _check(result):
            return [int(x) for x in result]
    
    # Try backtracking for 4 colors with different orderings
    if time.time() < deadline - 10:
        result = _try_colorings(4, deadline - 5, adj)
        if result is not None and _check(result):
            return [int(x) for x in result]
    
    # Fallback: greedy 4-coloring
    coloring = [-1] * n
    ordering = sorted(range(n), key=lambda x: -len(adj[x]))
    for v in ordering:
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        for c in range(4):
            if c not in used:
                coloring[v] = c
                break
    
    if -1 not in coloring and _check(coloring):
        return [int(x) for x in coloring]
    
    # Ultimate fallback: all distinct
    return [int(i) for i in range(n)]
