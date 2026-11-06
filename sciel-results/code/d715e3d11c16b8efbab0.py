import time
import random
import numpy as np

_T0 = time.time()
_BUDGET = 150.0
_MARGIN = 15.0


def _remaining():
    return _BUDGET - _MARGIN - (time.time() - _T0)


def _get_graph(n, p, seed):
    """Reproduce the verifier's random_graph(n, p, seed) exactly (matches seed program)."""
    rng = random.Random(seed)
    adj = np.zeros((n, n), dtype=bool)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u, v] = adj[v, u] = True
    return adj


def _maximal_cliques(adj, n):
    """Bron–Kerbosch with pivot, returns list of frozensets (maximal cliques)."""
    res = []
    P = [set(range(n)) for _ in range(n)]
    X = [set() for _ in range(n)]

    def bk(P, X):
        if not P:
            if X:
                return
            res.append(frozenset(P))
            return
        # pivot
        piv = -1
        best = -1
        for x in X:
            c = sum(1 for y in P if adj[x, y])
            if c > best:
                best = c
                piv = x
        for v in list(P):
            if piv >= 0 and not adj[v, piv]:
                continue
            Pn = [P[i] & ({v} | {w for w in P[i] if adj[v, w]}) for i in range(n)]
            Xn = [X[i] | ({v} if v in P[i] else set()) for i in range(n)]
            P.remove(v)
            bk(Pn, Xn)
            X.add(v)

    bk(P, X)
    return res


def _greedy_cover(adj, n, max_cliques):
    """Greedy set-cover: repeatedly pick the clique covering most uncovered edges."""
    mcs = [c for c in max_cliques if len(c) >= 2]
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.add((u, v))
    # precompute edge-set per clique
    ce = []
    for c in mcs:
        cs = sorted(c)
        e = frozenset((cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs)))
        ce.append(e)
    uncovered = set(edges)
    chosen = []
    while uncovered and _remaining() > 2.0:
        best = -1
        bestc = None
        for c, e in zip(mcs, ce):
            inter = len(e & uncovered)
            if inter > best:
                best = inter
                bestc = c
        if bestc is None:
            break
        chosen.append(bestc)
        uncovered -= ce[mcs.index(bestc)]
    # any leftover single edges (shouldn't happen)
    for e in uncovered:
        chosen.append(frozenset(e))
    return chosen


def _solve_ilp(adj, n, max_cliques, time_limit):
    """Exact min clique cover via MILP. Returns list of frozensets or None."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    mcs = [c for c in max_cliques if len(c) >= 2]
    m = len(mcs)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append((u, v))
    E = len(edges)
    if E == 0:
        return []
    A = np.zeros((E, m))
    for ci, c in enumerate(mcs):
        cs = sorted(c)
        idx = {v: k for k, v in enumerate(cs)}
        for ei, (u, v) in enumerate(edges):
            if u in idx and v in idx:
                A[ei, ci] = 1.0
    c_obj = np.ones(m)
    cons = LinearConstraint(A, np.ones(E), np.ones(E))
    bounds = Bounds(0.0, 1.0)
    res = milp(c_obj, constraints=cons, integrality=np.ones(m), bounds=bounds,
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 0:
        sol = res.x
        out = []
        for ci in range(m):
            if sol[ci] > 0.5:
                out.append(mcs[ci])
        return out
    return None


def _solve_instance(n, p, seed):
    adj = _get_graph(n, p, seed)
    if _remaining() <= 5.0:
        return _greedy_cover(adj, n, _maximal_cliques(adj, n))
    max_cliques = _maximal_cliques(adj, n)
    # try exact ILP with a bounded time
    tl = max(3.0, min(45.0, _remaining() - 8.0))
    exact = _solve_ilp(adj, n, max_cliques, tl)
    if exact is not None:
        return exact
    # fall back to greedy (always valid)
    return _greedy_cover(adj, n, max_cliques)


def construct():
    out = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() <= 3.0:
            adj = _get_graph(n, p, seed)
            out.append([list(c) for c in _greedy_cover(adj, n, _maximal_cliques(adj, n))])
        else:
            res = _solve_instance(n, p, seed)
            out.append([list(c) for c in res])
    return out
