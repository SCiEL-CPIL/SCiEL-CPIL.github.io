# EVOLVE-BLOCK-START
import time
import random

def _elapsed():
    return time.time() - _T0

_T0 = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill

def _random_graph(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve(n, p, seed, budget):
    edges = _random_graph(n, p, seed)
    m = len(edges)

    # adjacency as bitsets
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # edge -> index, and per-vertex incident edge lists
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # covered flag per edge
    covered = bytearray(m)

    # for each vertex, set of uncovered incident edges (as list of edge ids)
    # We'll maintain for fast "uncovered common neighborhood" queries.
    # uncovered_neighbors_bitset[v] = bitset of w such that edge (v,w) uncovered
    unc = [0] * n
    for i, (u, v) in enumerate(edges):
        unc[u] |= (1 << v)
        unc[v] |= (1 << u)

    # map edge id -> (u,v)
    eu = [0] * m
    ev = [0] * m
    for i, (u, v) in enumerate(edges):
        eu[i] = u
        ev[i] = v

    cliques = []

    # Precompute all-clique detection: a set S is a clique iff for all x in S, (S\{x}) subset of adj[x]
    def is_clique(S):
        for x in S:
            mask = 0
            for y in S:
                if y != x:
                    mask |= (1 << y)
            if (mask & ~adj[x]) != 0:
                return False
        return True

    # Greedy: repeatedly find the uncovered edge whose endpoints have the largest
    # common uncovered neighborhood, form the maximal clique, commit it.
    # To keep it fast, we sample candidate edges.
    uncovered_edges = set(range(m))
    rng = random.Random(seed + 777)

    while uncovered_edges:
        if _elapsed() > budget:
            break
        # pick candidate: try a few random uncovered edges, also a heuristic best
        # Find best candidate among a sample
        sample = list(uncovered_edges)
        rng.shuffle(sample)
        sample = sample[:min(len(sample), 40)]
        best_e = None
        best_score = -1
        best_clique = None
        for ei in sample:
            u, v = eu[ei], ev[ei]
            # common uncovered neighbors of u and v
            common = unc[u] & unc[v]
            # candidate clique = {u,v} + common (must be clique)
            # build clique greedily
            cand = [u, v]
            cset = {u, v}
            # add vertices from common that keep it a clique
            cm = common
            while cm:
                # pick a vertex
                lsb = cm & (-cm)
                w = lsb.bit_length() - 1
                cm ^= lsb
                # check w adjacent to all in cand
                ok = True
                for x in cand:
                    if not (adj[w] >> x) & 1:
                        ok = False
                        break
                if ok:
                    cand.append(w)
                    cset.add(w)
            # score = size
            if len(cand) > best_score:
                best_score = len(cand)
                best_e = ei
                best_clique = cand
        if best_e is None or best_score < 2:
            # fallback: take the edge as a 2-clique
            ei = next(iter(uncovered_edges))
            u, v = eu[ei], ev[ei]
            best_clique = [u, v]
        # commit best_clique: mark all its edges covered
        cl = best_clique
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                key = (a, b) if a < b else (b, a)
                eidx = edge_index[key]
                if not covered[eidx]:
                    covered[eidx] = 1
                    uncovered_edges.discard(eidx)
                    # update unc
                    unc[a] &= ~(1 << b)
                    unc[b] &= ~(1 << a)
        cliques.append(cl)

    # Local improvement: try to merge two cliques that share a vertex and together form a clique
    # (union is a clique) -> reduces count by 1.
    def try_merge(a, b):
        # a, b are cliques (lists). If union is a clique, merge.
        sa = set(a)
        sb = set(b)
        union = sa | sb
        # check union is a clique
        ul = list(union)
        for i in range(len(ul)):
            for j in range(i + 1, len(ul)):
                if not (adj[ul[i]] >> ul[j]) & 1:
                    return False
        return True

    # greedy merge pass
    changed = True
    while changed and _elapsed() < budget:
        changed = False
        for i in range(len(cliques)):
            for j in range(i + 1, len(cliques)):
                if _elapsed() > budget:
                    break
                a, b = cliques[i], cliques[j]
                # must share a vertex to be a valid single clique? No—union must be clique,
                # which requires all cross edges exist. They need not share a vertex.
                if try_merge(a, b):
                    cliques[i] = list(set(a) | set(b))
                    del cliques[j]
                    changed = True
                    break
            if changed:
                break

    return cliques

def construct():
    global _T0
    _T0 = time.time()
    deadline = _T0 + _DEADLINE
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        budget = deadline - 2.0  # leave margin
        if _elapsed() > budget - 1.0:
            # not enough time; produce trivial valid answer
            edges = _random_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        res = _solve(n, p, seed, budget)
        # ensure validity: every edge covered exactly once. Validate & repair.
        # Build coverage count
        from collections import defaultdict
        cov = defaultdict(int)
        for cl in res:
            if len(cl) < 2:
                continue
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    cov[(min(a, b), max(a, b))] += 1
        # edges of graph
        edges = _random_graph(n, p, seed)
        ok = True
        for (u, v) in edges:
            if cov[(u, v)] != 1:
                ok = False
                break
        # also ensure no extra edges covered (cliques must be cliques)
        if not ok:
            # fallback: trivial edge partition (always valid, each edge its own 2-clique)
            res = [list(e) for e in edges]
        results.append(res)
    return results
# EVOLVE-BLOCK-END
