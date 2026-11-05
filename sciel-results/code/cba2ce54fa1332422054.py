"""Improved clique partition: greedy max-clique selection with degree ordering."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 2.0 - (time.time() - _START)


def _generate_edges(n, p, seed):
    rng = random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _greedy_clique_partition(n, edges):
    edge_set = set(edges)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Order vertices by degree (desc) so high-degree vertices are processed first
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = {v: i for i, v in enumerate(order)}

    cliques = []
    used = set()

    # Process vertices in order; for each, greedily build the largest clique
    # from its still-unused neighbors, in order of descending remaining degree.
    for v in order:
        if v in used:
            continue
        # Candidate neighbors: adjacent, unused
        cand = [u for u in adj[v] if u not in used]
        if not cand:
            continue
        # Sort candidates by degree (desc) to prefer dense regions
        cand.sort(key=lambda x: -len(adj[x]))
        clique = [v]
        clique_set = {v}
        for u in cand:
            if u in used:
                continue
            # u must be adjacent to every member of the current clique
            if all(w in adj[u] for w in clique):
                clique.append(u)
                clique_set.add(u)
        # The clique has at least 2 vertices (v and at least one neighbor)
        # Mark all its edges as used
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                a, b = clique[i], clique[j]
                e = (a, b) if a < b else (b, a)
                used.add(e)
        cliques.append(clique)
        used.update(clique_set)

    # Safety: cover any remaining edges (should be none for a valid partition)
    for e in edge_set:
        if e not in used:
            cliques.append(list(e))
            used.add(e)

    return cliques


def _partition_instance(n, p, seed):
    edges = _generate_edges(n, p, seed)
    return _greedy_clique_partition(n, edges)


def construct():
    r1 = _partition_instance(32, 0.6, 601)
    if _remaining() < 10.0:
        # Not enough time; fall back to trivial edges for instance 2
        edges2 = _generate_edges(64, 0.55, 602)
        r2 = [list(e) for e in edges2]
        return [r1, r2]
    r2 = _partition_instance(64, 0.55, 602)
    return [r1, r2]
# EVOLVE-BLOCK-END
