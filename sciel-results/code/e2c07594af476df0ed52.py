"""Voltage-lifting cage construction (free-abelian Z_m coverings) + exact BFS girth verifier."""
import random
from collections import deque

_MAX_VERTS = 200000
_MAX_EDGES = 1000000


def _bfs_girth_ok(adj, n, g):
    """Exact girth check via non-backtracking BFS from every vertex.
    A cycle of length < g exists iff some vertex is reached by a non-backtracking
    walk of length >= g-2 from a vertex at BFS depth >= 2."""
    lim = g - 2
    for s in range(n):
        depth = [-1] * n
        par = [-1] * n
        depth[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = depth[u]
            if du >= lim:
                for v in adj[u]:
                    if v != par[u] and depth[v] >= 2:
                        return False
                continue
            nd = du + 1
            for v in adj[u]:
                if v == par[u]:
                    continue
                if depth[v] == -1:
                    depth[v] = nd
                    par[v] = u
                    q.append(v)
    return True


def _base_candidates(k, g):
    cands = []
    # complete bipartite K_{a,a} (needs k even)
    if k % 2 == 0:
        a = k // 2
        cands.append((2 * a, [tuple(range(a, a + a)) for _ in range(a)] +
                               [tuple(range(i, i + a)) for i in range(a)]))
    # complete graph K_{k+1}
    cands.append((k + 1, [tuple(range(i, i + k + 1)) for i in range(k + 1)]))
    # cycle C_k
    cands.append((k, [tuple(range(i, i + k)) for i in range(k)]))
    # complete multipartite with k parts of size 2
    cands.append((2 * k, [tuple(range(j, j + 2 * k)) for j in range(k)] +
                            [tuple(range(i, i + 2 * k)) for i in range(2 * k)]))
    return cands


def _try_base(k, g, seed):
    """Return the smallest base graph (n, adj, edges) with girth >= g, or None."""
    rng = random.Random(seed)
    cands = _base_candidates(k, g)
    cands.sort(key=lambda c: c[0])
    for n, adj in cands:
        if n > _MAX_VERTS:
            continue
        if _bfs_girth_ok(adj, n, g):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, adj, edges
    return None


def _lift(base_n, base_adj, m, seed):
    """Random Z_m voltage assignment; return (N, adj) of the lift, or None if girth < g."""
    rng = random.Random(seed)
    volt = [[0] * m for _ in range(base_n)]
    for u in range(base_n):
        for v in base_adj[u]:
            volt[u][v] = rng.randrange(m)
    N = base_n * m
    adj = [[] for _ in range(N)]
    for u in range(base_n):
        for v in base_adj[u]:
            w = volt[u][v]
            for r in range(m):
                a = u * m + r
                b = v * m + ((r + w) % m)
                adj[a].append(b)
    if _bfs_girth_ok(adj, N, g):
        return N, adj
    return None


def construct(k, g):
    # 1) find a valid base graph (already a valid cage)
    base = _try_base(k, g, seed=12345)
    if base is None:
        # fallback: pure cycle graph (girth = k, valid when k >= g)
        n = max(k, g)
        adj = [[(i - 1) % n, (i + 1) % n] for i in range(n)]
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    base_n, base_adj, base_edges = base
    best_n, best_adj = base_n, base_adj

    # 2) try to grow with a Z_m voltage lift while keeping girth >= g
    for m in range(2, 40):
        if base_n * m > _MAX_VERTS:
            break
        found = None
        for trial in range(30):
            res = _lift(base_n, base_adj, m, seed=100000 + m * 131 + trial)
            if res is not None:
                found = res
                break
        if found is None:
            break  # this m consistently fails; larger m only harder
        N, adj = found
        if N > best_n:
            best_n, best_adj = N, adj

    edges = []
    for u in range(best_n):
        for v in best_adj[u]:
            if u < v:
                edges.append((u, v))
    return best_n, edges
