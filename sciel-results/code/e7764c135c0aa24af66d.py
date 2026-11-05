"""Clique partitioning of two fixed dense random graphs."""
# EVOLVE-BLOCK-START
import time

def _generate_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _maximal_cliques(adj, n):
    """Bron-Kerbosch with pivot, returns maximal cliques (each as a set)."""
    cliques = []

    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Pivot: choose u in P union X with max |P & adj[u]|
        PnX = P | X
        if PnX:
            best, bestc = None, -1
            for u in PnX:
                c = len(P & adj[u])
                if c > bestc:
                    bestc, best = c, u
            candidates = P - adj[best]
        else:
            candidates = P
        for v in list(candidates):
            P.remove(v)
            bk(R | {v}, P & adj[v], X & adj[v])
            X.add(v)

    bk(frozenset(), set(range(n)), frozenset())
    return cliques


def _partition_one(n, p, seed, deadline):
    edges = _generate_edges(n, p, seed)
    adj = _build_adjacency(edges, n)
    edge_set = set(edges)
    total = len(edge_set)

    # Fallback: each edge as its own clique (always valid, score baseline)
    fallback = [[u, v] for (u, v) in edges]

    try:
        cliques = _maximal_cliques(adj, n)
    except Exception:
        return fallback

    # Greedy set cover over maximal cliques by uncovered-edge count
    uncovered = set(edge_set)
    chosen = []
    for c in cliques:
        if time.time() > deadline:
            break
        if len(c) < 2:
            continue
        covered = 0
        for u in c:
            for v in c:
                if u < v:
                    if (u, v) in uncovered:
                        covered += 1
        if covered > 0:
            chosen.append(c)
            for u in c:
                for v in c:
                    if u < v:
                        uncovered.discard((u, v))
        if not uncovered:
            break

    # Cover any leftover edges individually (guarantees validity)
    for e in list(uncovered):
        chosen.append(frozenset(e))

    if not chosen:
        return fallback
    return [sorted(list(c)) for c in chosen]


def construct():
    deadline = time.time() + 120.0
    out = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        out.append(_partition_one(n, p, seed, deadline))
    return out
# EVOLVE-BLOCK-END
