"""Clique-partition of two fixed dense random graphs.

Strategy:
  1. Build the graph and its adjacency as bitmasks for fast intersection.
  2. Greedily extract maximal cliques, always preferring the largest clique
     available (most edges removed per clique).  Each extracted clique
     absorbs every unassigned edge inside it.
  3. After the greedy pass, run a bounded polish pass that repeatedly tries
     to merge two cliques whose union is still a clique (absorbing any edges
     between them), which strictly reduces the number of cliques.
  4. A wall-clock deadline guard returns the best partition found so far.
"""

import time

# ---- task parameters (fixed by the verifier) ----
_PARAMS = ((32, 0.6, 601), (64, 0.55, 602))
_DEADLINE = 140.0   # seconds of wall-clock to stay safely under the 150s kill


def _build_graph(n, p, seed):
    """Return (adjacency bitmasks, edge list) for random_graph(n, p, seed)."""
    from random import Random
    rng = Random(seed)
    adj = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
    return adj, edges


def _greedy_max_clique(adj, n, start, max_size=6):
    """Return a maximal clique (list of vertices) containing start.

    Uses the standard "intersect common neighbours" branch-and-bound idea but
    stops as soon as we have a reasonably large clique (max_size) to keep the
    per-clique cost low.  The returned clique is maximal w.r.t. the current
    adjacency (no outside vertex is adjacent to all members).
    """
    clique = [start]
    # candidates that can be added: neighbours of every current member
    cand = adj[start]
    for _ in range(max_size - 1):
        if cand == 0:
            break
        # pick the candidate vertex with the most neighbours among cand
        best = None
        best_deg = -1
        c = cand
        while c:
            low = c & (-c)
            v = low.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best = v
            c &= c - 1
        if best is None:
            break
        clique.append(best)
        cand &= adj[best]
    # try to extend with any remaining common neighbours up to a cap
    c = cand
    added = 0
    while c and added < (max_size - len(clique)):
        low = c & (-c)
        v = low.bit_length() - 1
        clique.append(v)
        cand &= adj[v]
        added += 1
    return clique


def _solve_instance(n, p, seed, deadline):
    adj, edges = _build_graph(n, p, seed)

    # map each edge to an index; track which are still unassigned
    edge_set = set(edges)
    unassigned = set(edges)

    # for fast "is vertex in clique" and membership tests
    cliques = []
    vertex_used = [0] * n  # bitmask of cliques each vertex belongs to (for polish)

    # precompute per-vertex neighbour list for polish checks
    neigh = [0] * n  # bitmasks

    def clique_is_valid(vertices):
        # every pair must be an edge
        for i in range(len(vertices)):
            a = vertices[i]
            for j in range(i + 1, len(vertices)):
                b = vertices[j]
                if not (adj[a] >> b) & 1:
                    return False
        return True

    while unassigned:
        if time.time() > deadline:
            break
        # pick the edge with the most "clique potential": choose an edge (u,v)
        # whose common-neighbour count is largest, to get a big clique.
        best_edge = None
        best_common = -1
        # sample a subset if many edges remain to keep cost bounded
        items = list(unassigned)
        if len(items) > 400:
            import random as _r
            _r.seed(0)
            items = _r.sample(items, 400)
        for (u, v) in items:
            common = (adj[u] & adj[v]).bit_count()
            if common > best_common:
                best_common = common
                best_edge = (u, v)
        if best_edge is None:
            best_edge = next(iter(unassigned))

        u0, v0 = best_edge
        # build a maximal clique starting from u0
        clique = _greedy_max_clique(adj, n, u0)
        # ensure v0 is in it if possible (it's a neighbour of u0)
        if v0 not in clique:
            # try to extend with v0 if it connects to all
            ok = all((adj[v0] >> x) & 1 for x in clique)
            if ok:
                clique.append(v0)
        # absorb all unassigned edges inside this clique
        absorbed = set()
        for i in range(len(clique)):
            a = clique[i]
            for j in range(i + 1, len(clique)):
                b = clique[j]
                e = (a, b) if a < b else (b, a)
                if e in unassigned:
                    absorbed.add(e)
        if not absorbed:
            # fallback: take just the single edge
            absorbed = {best_edge}
        unassigned -= absorbed
        if len(clique) >= 2:
            cliques.append(sorted(clique))
        else:
            cliques.append(sorted([best_edge[0], best_edge[1]]))

    # ---- polish pass: merge cliques whose union is a clique ----
    def try_merge(cliques, deadline):
        merged = True
        while merged and time.time() < deadline:
            merged = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    a = set(cliques[i])
                    b = set(cliques[j])
                    union = a | b
                    if len(union) < 2:
                        continue
                    if clique_is_valid(list(union)):
                        # merge i and j, and absorb any edges between them
                        newclique = sorted(union)
                        # collect all edges within newclique that are still
                        # represented by a separate clique and absorb them
                        cliques = [c for k, c in enumerate(cliques)
                                   if k != i and k != j]
                        cliques.append(newclique)
                        merged = True
                        break
                if merged:
                    break
        return cliques

    cliques = try_merge(cliques, deadline)

    # ---- validate and repair: any edge not covered must become its own clique ----
    covered = set()
    for c in cliques:
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                a, b = c[i], c[j]
                e = (a, b) if a < b else (b, a)
                covered.add(e)
    # every edge must be covered exactly once; if an edge is missing add it
    for e in edges:
        if e not in covered:
            cliques.append(sorted(e))
            covered.add(e)

    # remove any clique with <2 vertices (shouldn't happen)
    cliques = [c for c in cliques if len(c) >= 2]
    return cliques


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    results = []
    for (n, p, seed) in _PARAMS:
        if time.time() > deadline:
            # not enough time; fall back to single-edge cliques for remaining
            adj, edges = _build_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        # give each instance its own sub-deadline so we finish both
        sub_deadline = min(deadline, t0 + _DEADLINE - 1.0)
        results.append(_solve_instance(n, p, seed, sub_deadline))
    return results
