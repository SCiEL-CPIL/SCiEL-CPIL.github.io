"""Decompose three fixed graphs into few cycles/edges via max 2-matching (bipartite double cover)."""
import time
from random import Random

_DEADLINE = 115.0  # safety margin under the 150s wall-clock kill


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _max_flow(n, cap):
    """Dinic max-flow on a directed graph; returns (flow, parent)."""
    adj = [[] for _ in range(n)]
    idx = [0] * n

    def add(u, v, c):
        adj[u].append(v)
        cap[u].append(c)
        adj[v].append(u)
        cap[v].append(0)
        idx[u] += 1

    flow = 0
    INF = 1 << 60
    while True:
        level = [-1] * n
        q = [0]
        level[0] = 0
        while q:
            u = q.pop(0)
            i = 0
            while i < len(adj[u]):
                v = adj[u][i]
                if cap[u][i] > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
                i += 1
        if level[n - 1] < 0:
            break
        it = [0] * n

        def dfs(u, f):
            if u == n - 1:
                return f
            i = it[u]
            while i < len(adj[u]):
                v = adj[u][i]
                if cap[u][i] > 0 and level[v] == level[u] + 1:
                    ret = dfs(v, f if f < cap[u][i] else cap[u][i])
                    if ret > 0:
                        cap[u][i] -= ret
                        cap[v][i + 1] += ret
                        return ret
                i += 1
            it[u] = i
            return 0

        while True:
            pushed = dfs(0, INF)
            if pushed == 0:
                break
            flow += pushed
    parent = {}
    for u in range(n):
        i = 0
        while i < len(adj[u]):
            v = adj[u][i]
            if i % 2 == 0 and cap[u][i] < 1:
                parent[v] = u
            i += 1
    return flow, parent


def _max_2matching(edges, n):
    """Max edge count of a subgraph with degree<=2 via bipartite double cover.
    Returns the set of undirected edges in the 2-matching."""
    if not edges:
        return set()
    N = 2 * n + 2
    S = 2 * n
    T = 2 * n + 1
    cap = [0] * (N * N)

    def add(u, v, c):
        cap[u * N + v] += c
        cap[v * N + u] += 0

    # source -> left copies
    for u in range(n):
        add(S, u, 2)
    # right copies -> sink
    for v in range(n):
        add(v + n, T, 2)
    # for each undirected edge (a,b): aL->bR and bL->aR
    for (a, b) in edges:
        add(a, b + n, 1)
        add(b, a + n, 1)

    flow, parent = _max_flow(N, cap)
    # parent maps node -> predecessor; predecessor of a right node is a left node.
    sel = set()
    for v in range(n):
        rn = v + n
        if rn in parent:
            ln = parent[rn]
            if ln < n:
                sel.add(tuple(sorted((ln, v))))
    return sel


def _extract_cycles(mset):
    """From a 2-matching edge set, greedily remove longest cycles first.
    Returns (cycles, remaining_edges)."""
    adj = {}
    for (a, b) in mset:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    cycles = []
    while True:
        best = None
        for s in adj:
            if not adj[s]:
                continue
            for nb in list(adj[s]):
                if nb == s:
                    continue
                # walk a cycle starting s -> nb
                path = [s, nb]
                prev, cur = s, nb
                ok = True
                while cur != s:
                    nxts = [x for x in adj[cur] if x != prev]
                    if len(nxts) == 0:
                        ok = False
                        break
                    nxt = nxts[0]
                    if nxt == s:
                        break
                    if nxt in path:
                        ok = False
                        break
                    path.append(nxt)
                    prev, cur = cur, nxt
                if ok and path[-1] != s and len(path) >= 3:
                    if best is None or len(path) > len(best):
                        best = path
        if best is None:
            break
        for i in range(len(best)):
            a, b = best[i], best[(i + 1) % len(best)]
            e = tuple(sorted((a, b)))
            mset.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        cycles.append(best)
    return cycles, list(mset)


def _decompose(edges, n):
    mset = _max_2matching(edges, n)
    cycles, rest = _extract_cycles(mset)
    pieces = [list(c) for c in cycles]
    for e in rest:
        pieces.append(list(e))
    return pieces


def construct():
    t0 = time.time()
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > _DEADLINE:
            # safety: fall back to trivial edge decomposition for remaining
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        n = spec[0]
        try:
            pieces = _decompose(edges, n)
        except Exception:
            pieces = [list(e) for e in edges]
        results.append(pieces)
    return results
