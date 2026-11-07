"""Direct construction of a 4-coloring of the degree-4 grid graph
that is adjacent-different and has no bicolored cycle (i.e. every
two-color induced subgraph is a forest)."""
import time

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
    # For each color pair, union-find over induced subgraph
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # We need per-pair checks; do them independently
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                c1, c2 = coloring[u], coloring[v]
                pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))

    for (a, b), edges in pair_edges.items():
        # Only vertices involved
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        # Union-find on these verts
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


def _backtrack(adj, nc, deadline, init_coloring=None):
    n = P['n']
    if init_coloring is None:
        coloring = [-1] * n
    else:
        coloring = init_coloring[:]
        
    uf_states = {}
    
    def find(x, par):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def dfs():
        if time.time() > deadline:
            return None
        
        min_count = nc + 1
        best_v = -1
        best_colors = []
        
        for v in range(n):
            if coloring[v] == -1:
                used = set()
                for nb in adj[v]:
                    if coloring[nb] != -1:
                        used.add(coloring[nb])
                avail = [c for c in range(nc) if c not in used]
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
            
            affected_pairs = set()
            for nb in adj[v]:
                cn = coloring[nb]
                if cn != -1 and cn != c:
                    affected_pairs.add((min(c, cn), max(c, cn)))
            
            saved_states = {}
            for p in affected_pairs:
                if p in uf_states:
                    saved_states[p] = (uf_states[p][0][:], uf_states[p][1][:])
            
            for p in affected_pairs:
                if p not in uf_states:
                    uf_states[p] = (list(range(n)), [0]*n)
            
            ok = True
            for nb in adj[v]:
                cn = coloring[nb]
                if cn != -1 and cn != c:
                    p = (min(c, cn), max(c, cn))
                    par, rn = uf_states[p]
                    rx = find(v, par)
                    ry = find(nb, par)
                    if rx == ry:
                        ok = False
                        break
                    if rn[rx] < rn[ry]:
                        rx, ry = ry, rx
                    par[ry] = rx
                    if rn[rx] == rn[ry]:
                        rn[rx] += 1
            
            if ok:
                res = dfs()
                if res is not None:
                    return res
            
            for p in affected_pairs:
                if p in saved_states:
                    uf_states[p] = saved_states[p]
                else:
                    if p in uf_states:
                        del uf_states[p]
            
            coloring[v] = -1
        return None

    return dfs()


def construct():
    start_time = time.time()
    deadline = start_time + 80
    adj = _build_adj()

    # Try 3 colors first, then 4
    for nc in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        
        init = [-1] * P['n']
        init[0] = 0
        
        result = _backtrack(adj, nc, deadline, init)
        if result is not None:
            if _check(result):
                return [int(x) for x in result]

    # Fallback: all distinct colors
    return [int(i) for i in range(P['n'])]
