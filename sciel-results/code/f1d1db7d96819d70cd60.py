import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _girth_bfs(adj, n, limit):
    """Smallest cycle length in an undirected simple graph, via multi-source BFS.
    Returns limit if no cycle shorter than limit is found."""
    if limit <= 2:
        return limit
    dist = [-1] * n
    par = [-1] * n
    q = deque()
    for s in range(n):
        dist[s] = 0
        q.append(s)
    best = limit
    while q:
        u = q.popleft()
        du = dist[u]
        if du + 1 >= best:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                par[v] = u
                q.append(v)
            elif v > u and par[v] != u and par[u] != v:
                c = du + dist[v] + 1
                if c < best:
                    best = c
                    if best <= 2:
                        return 2
    return best

def _find_short_cycles(adj, n, g, max_cycles=200):
    """Find cycles of length < g. Returns list of edge-index lists."""
    found = []
    if g <= 4:
        return found
    depth = g // 2  # half-length to search
    for s in range(n):
        # BFS from s tracking parent edge
        dist = [-1] * n
        pedge = [-1] * n
        q = deque()
        dist[s] = 0
        q.append(s)
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= depth:
                continue
            for ei, v in enumerate(adj[u]):
                if dist[v] == -1:
                    dist[v] = du + 1
                    pedge[v] = (u, ei)
                    q.append(v)
                elif v > s and pedge[v] != (u, ei) and pedge[u] != (v, ei):
                    c = du + dist[v] + 1
                    if c < g:
                        # reconstruct cycle
                        cycle = []
                        # path u -> s
                        a = u
                        while a != s:
                            pu, pei = pedge[a]
                            cycle.append((pu, pei))
                            a = pu
                        # path v -> s
                        b = v
                        while b != s:
                            pb, pei = pedge[b]
                            cycle.append((pb, pei))
                            b = pb
                        if len(cycle) == c:
                            found.append(cycle)
                        if len(found) >= max_cycles:
                            return found
    return found

def construct(k, g):
    t_start = time.time()
    deadline = t_start + 112.0
    rng = random.Random(20260101)

    if k != 4:
        # Fallback to LZW construction for other degrees
        return _lzw_fallback(k, g)

    # Target: 4-regular bipartite graph on n vertices (n even), girth >= g.
    # Start from tensor product of two cycles C_L x C_L.
    n = 244  # even, above Moore bound 243, small enough for fast girth check
    L = n // 2  # 122

    # Build tensor product: vertex (i,j) -> i*L + j, edges to (i+1,j),(i-1,j),(i,j+1),(i,j-1) mod L
    adj = [set() for _ in range(n)]
    edges = []
    edge_set = set()
    for i in range(L):
        for j in range(L):
            v = i * L + j
            for ni, nj in ((i + 1) % L, j), ((i - 1) % L, j), (i, (j + 1) % L), (i, (j - 1) % L):
                w = ni * L + nj
                if v < w:
                    key = (v, w)
                    if key not in edge_set:
                        edge_set.add(key)
                        edges.append(key)
                        adj[v].add(w)
                        adj[w].add(v)

    # Verify initial girth
    g0 = _girth_bfs(adj, n, g)
    if g0 >= g:
        return n, edges

    # Local search: 2-opt edge swaps to eliminate short cycles.
    # A swap: remove edges (a,b) and (c,d), add (a,c) and (b,d),
    # where a,b,c,d are distinct and the new edges don't already exist.
    best_n = n
    best_edges = edges[:]
    best_g = g0

    def try_swaps(max_iters):
        nonlocal adj, edges, edge_set, best_edges, best_g
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                return
            # Find a short cycle to target
            cycles = _find_short_cycles(adj, n, g, max_cycles=30)
            if not cycles:
                # No short cycles found by cycle finder; verify with BFS
                g_now = _girth_bfs(adj, n, g)
                if g_now >= g:
                    best_edges = edges[:]
                    best_g = g_now
                    return
                # Cycle finder missed it; do random swaps
                cycles = None
            if cycles:
                # Pick a random edge from a random short cycle
                cyc = rng.choice(cycles)
                ei = rng.randrange(len(cyc))
                u, idx = cyc[ei]
                # The edge is (u, adj[u][idx])
                v = list(adj[u])[idx]
                # Find another edge to swap with: pick a random edge not in cycle
                # For speed, pick a random vertex and one of its edges
                for _ in range(20):
                    c = rng.randrange(n)
                    if c == u or c == v:
                        continue
                    cand = list(adj[c])
                    if not cand:
                        continue
                    d = rng.choice(cand)
                    if d == u or d == v:
                        continue
                    # Check new edges (u,c) and (v,d) don't exist and aren't loops
                    if c in adj[u] or d in adj[u] or c in adj[v] or d in adj[v]:
                        continue
                    if c == u or d == v or c == v or d == u:
                        continue
                    # Perform swap
                    # Remove (u,v) and (c,d)
                    adj[u].discard(v); adj[v].discard(u)
                    adj[c].discard(d); adj[d].discard(c)
                    edge_set.discard((u, v) if u < v else (v, u))
                    edge_set.discard((c, d) if c < d else (d, c))
                    edges = [e for e in edges if e != (u, v) and e != (v, u) and e != (c, d) and e != (d, c)]
                    # Add (u,c) and (v,d)
                    adj[u].add(c); adj[c].add(u)
                    adj[v].add(d); adj[d].add(v)
                    key1 = (u, c) if u < c else (c, u)
                    key2 = (v, d) if v < d else (d, v)
                    edge_set.add(key1); edge_set.add(key2)
                    edges.append(key1); edges.append(key2)
                    break
            else:
                # Random swap
                for _ in range(20):
                    a = rng.randrange(n)
                    b = rng.choice(list(adj[a]))
                    c = rng.randrange(n)
                    if c == a or c == b:
                        continue
                    d = rng.choice(list(adj[c]))
                    if d == a or d == b:
                        continue
                    if c in adj[a] or d in adj[a] or c in adj[b] or d in adj[b]:
                        continue
                    # Perform swap
                    adj[a].discard(b); adj[b].discard(a)
                    adj[c].discard(d); adj[d].discard(c)
                    edge_set.discard((a, b) if a < b else (b, a))
                    edge_set.discard((c, d) if c < d else (d, c))
                    edges = [e for e in edges if e != (a, b) and e != (b, a) and e != (c, d) and e != (d, c)]
                    adj[a].add(c); adj[c].add(a)
                    adj[b].add(d); adj[d].add(b)
                    key1 = (a, c) if a < c else (c, a)
                    key2 = (b, d) if b < d else (d, b)
                    edge_set.add(key1); edge_set.add(key2)
                    edges.append(key1); edges.append(key2)
                    break

    # Phase 1: targeted search for short cycles
    for phase in range(8):
        if time.time() > deadline - 10.0:
            break
        try_swaps(3000)
        g_now = _girth_bfs(adj, n, g)
        if g_now >= g:
            best_edges = edges[:]
            best_g = g_now
            break
        if g_now < best_g:
            best_g = g_now
            best_edges = edges[:]

    # Phase 2: random restarts if still not girth g
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        # Rebuild from tensor product with random perturbation
        adj = [set() for _ in range(n)]
        edges = []
        edge_set = set()
        for i in range(L):
            for j in range(L):
                v = i * L + j
                for ni, nj in ((i + 1) % L, j), ((i - 1) % L, j), (i, (j + 1) % L), (i, (j - 1) % L):
                    w = ni * L + nj
                    if v < w:
                        key = (v, w)
                        if key not in edge_set:
                            edge_set.add(key)
                            edges.append(key)
                            adj[v].add(w)
                            adj[w].add(v)
        # Random perturbation: do some random swaps
        for _ in range(500):
            a = rng.randrange(n)
            b = rng.choice(list(adj[a]))
            c = rng.randrange(n)
            if c == a or c == b:
                continue
            d = rng.choice(list(adj[c]))
            if d == a or d == b:
                continue
            if c in adj[a] or d in adj[a] or c in adj[b] or d in adj[b]:
                continue
            adj[a].discard(b); adj[b].discard(a)
            adj[c].discard(d); adj[d].discard(c)
            edge_set.discard((a, b) if a < b else (b, a))
            edge_set.discard((c, d) if c < d else (d, c))
            edges = [e for e in edges if e != (a, b) and e != (b, a) and e != (c, d) and e != (d, c)]
            adj[a].add(c); adj[c].add(a)
            adj[b].add(d); adj[d].add(b)
            key1 = (a, c) if a < c else (c, a)
            key2 = (b, d) if b < d else (d, b)
            edge_set.add(key1); edge_set.add(key2)
            edges.append(key1); edges.append(key2)
        # Local search
        try_swaps(5000)
        g_now = _girth_bfs(adj, n, g)
        if g_now >= g:
            best_edges = edges[:]
            best_g = g_now
            break
        if g_now > best_g:
            best_g = g_now
            best_edges = edges[:]

    return n, best_edges

def _lzw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar construction for general k."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
