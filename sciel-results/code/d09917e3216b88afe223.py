"""Clique partition via greedy maximal-clique packing."""
import time
from itertools import combinations

_DEADLINE = 120.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _make_graph(n, p, seed):
    import random
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    return adj, edges

def _solve(n, p, seed):
    adj, edges = _make_graph(n, p, seed)
    # adjacency as bitmasks for fast intersection
    mask = [0] * n
    for u in range(n):
        m = 0
        for v in adj[u]:
            m |= (1 << v)
        mask[u] = m
    # edge presence mask
    present = [True] * len(edges)
    edge_set = set(edges)
    # map edge -> index
    idx = {e: i for i, e in enumerate(edges)}

    # We'll maintain for each edge a "clique potential" = size of common neighbors.
    # Greedily: pick the edge whose endpoints have the largest intersection of
    # still-present neighbors, build maximal clique, remove its edges.

    # To speed up, precompute per-vertex adjacency list.
    adj_list = [sorted(adj[u]) for u in range(n)]

    # current active edges as a set
    active = set(edges)

    cliques = []

    while active and _remaining() > 1.5:
        # Score each active edge by |common active neighbors|
        # common neighbors of (u,v) = adj[u] & adj[v] that are still connected
        best_edge = None
        best_common = None
        best_score = -1
        for (u, v) in active:
            common = adj[u] & adj[v]
            # only count neighbors that still have an edge to u or v? For clique
            # extension we need the vertex to connect to ALL clique members.
            # Start clique = {u, v}; can add w if w connects to both u and v.
            # That's exactly adj[u] & adj[v].
            score = len(common)
            if score > best_score:
                best_score = score
                best_edge = (u, v)
                best_common = common
        if best_edge is None:
            break
        u, v = best_edge
        # Build maximal clique starting from {u, v}
        clique = {u, v}
        # candidate vertices: common neighbors
        cand = set(best_common)
        # Greedily add vertices that connect to all current clique members
        # To be a clique, every new vertex must connect to every existing member.
        # We can iteratively add; order by degree within cand for larger cliques.
        # Compute for each candidate its connection to all clique members.
        while cand:
            # pick candidate with max connections to current clique & among cand
            # A candidate w is addable if w in adj[x] for all x in clique.
            addable = [w for w in cand if all(w in adj[x] for x in clique)]
            if not addable:
                break
            # choose the one that keeps the most options: max |adj[w] & addable & clique-adj|
            def wscore(w):
                # number of other addable vertices that also connect to w
                return sum(1 for z in addable if z != w and z in adj[w])
            # sort by score desc, break ties by vertex id
            addable.sort(key=lambda w: (-wscore(w), w))
            w = addable[0]
            clique.add(w)
            # restrict candidates: must connect to w too
            cand = cand & adj[w]
        # Convert to list
        cl = sorted(clique)
        cliques.append(cl)
        # Remove all edges within this clique from active
        for a, b in combinations(cl, 2):
            active.discard((a, b))
            active.discard((b, a))

    # Any remaining edges must each be their own clique (they are edges, size 2)
    for (u, v) in active:
        cliques.append([u, v])

    return cliques

def construct():
    global _START
    _START = time.time()
    r1 = _solve(32, 0.6, 601)
    r2 = _solve(64, 0.55, 602)
    return [r1, r2]
