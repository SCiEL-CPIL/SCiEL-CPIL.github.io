"""Rewritten program: minimum vertex cover via branch-and-bound + greedy fallback."""
# EVOLVE-BLOCK-START
import time
from random import Random

_DEADLINE = 150.0 - 8.0   # safety margin before the 150s wall-clock kill


def _build_graph(n, p, seed):
    """Reproduce random_graph(n,p,seed) edges exactly as in the seed program."""
    rng = Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return edges, adj


def _maximal_cliques(adj, n):
    """Bron–Kerbosch with pivot; returns list of maximal cliques (frozensets)."""
    cliques = []

    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # pivot: choose vertex in P ∪ X with max |P ∩ N(u)|
        cand = P | X
        best = -1
        pivot = None
        for u in cand:
            c = len(P & adj[u])
            if c > best:
                best = c
                pivot = u
        for v in list(P):
            if pivot is not None and v in adj[pivot]:
                continue
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    bronk(frozenset(), set(range(n)), set())
    return cliques


def _greedy_cover(edges, adj, n):
    """Standard greedy vertex cover: repeatedly pick vertex covering most uncovered edges."""
    uncovered = set(edges)
    cover = set()
    while uncovered:
        best_v = None
        best_c = -1
        for v in range(n):
            c = 0
            for w in adj[v]:
                e = (v, w) if v < w else (w, v)
                if e in uncovered:
                    c += 1
            if c > best_c:
                best_c = c
                best_v = v
                if c == len(uncovered):
                    break
        if best_c <= 0:
            break
        cover.add(best_v)
        for w in adj[best_v]:
            e = (best_v, w) if best_v < w else (w, best_v)
            uncovered.discard(e)
    return cover


def _min_vertex_cover(adj, n, deadline):
    """Exact min vertex cover = n - max independent set, via branch & bound on complement."""
    # complement adjacency
    comp = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and v not in adj[u]:
                comp[u].add(v)

    best = [0]
    best_set = [set()]
    order = sorted(range(n), key=lambda v: -len(comp[v]))

    def expand(I, cand, idx):
        if idx == n or len(I) + len(cand) <= best[0]:
            if len(I) > best[0]:
                best[0] = len(I)
                best_set[0] = set(I)
            return
        if time.time() > deadline:
            return
        # try each remaining candidate
        i = idx
        while i < n:
            v = order[i]
            if v in cand:
                newI = I | {v}
                newCand = cand & comp[v]
                expand(newI, newCand, i + 1)
                cand.discard(v)
            i += 1

    expand(set(), set(range(n)), 0)
    # vertex cover = all vertices minus max independent set
    cover = set(range(n)) - best_set[0]
    return cover, best[0]


def _valid(cover, cliques):
    """Check that every maximal clique (size>=2) contains at least one cover vertex."""
    for c in cliques:
        if not (c & cover):
            return False
    return True


def construct():
    t_start = time.time()
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges, adj = _build_graph(n, p, seed)
        cliques = _maximal_cliques(adj, n)

        # exact solver with a per-instance time cap
        per_inst = min(30.0, max(2.0, _DEADLINE - (time.time() - t_start)))
        try:
            cover, _ = _min_vertex_cover(adj, n, time.time() + per_inst)
            if _valid(cover, cliques):
                results.append(sorted(cover))
                continue
        except Exception:
            pass

        # fallback: greedy (always valid as a vertex cover, hence hits every clique)
        cover = _greedy_cover(edges, adj, n)
        if not _valid(cover, cliques):
            # last resort: all non-isolated vertices
            cover = {x for e in edges for x in e}
        results.append(sorted(cover))
    return results
# EVOLVE-BLOCK-END
