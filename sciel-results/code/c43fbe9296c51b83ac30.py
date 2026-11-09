"""Improved program: cover edges of each split graph with a minimum clique cover."""
import time

_DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150s)


def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        # Regenerate the split graph and build the cross-adjacency structure.
        rng = Random(seed)
        adj = [0] * k  # adj[c] = bitmask of independent vertices (0..n-k-1) adjacent to clique vertex c
        for v in range(k, n):
            iv = v - k
            bit = 1 << iv
            for c in range(k):
                if rng.random() < p:
                    adj[c] |= bit

        cliques = []
        # 1) All edges inside the clique part are covered by the single clique C.
        cliques.append(list(range(k)))
        # 2) Every cross edge (c, iv) is covered by the star clique {c} ∪ S_c,
        #    where S_c is the (maximal) set of independent vertices adjacent to c.
        #    A vertex iv appears in exactly one S_c (per adjacent c), so each cross
        #    edge lies in exactly one clique. No independent vertex is isolated,
        #    so every listed clique has at least two vertices.
        for c in range(k):
            mask = adj[c]
            if mask:
                s = [c]
                iv = 0
                while mask:
                    if mask & 1:
                        s.append(k + iv)
                    mask >>= 1
                    iv += 1
                cliques.append(s)
        return cliques

    results = []
    start = time.time()
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
        if time.time() > _DEADLINE - 5.0:
            break

    # Enforce exact types: Python int vertices, list/tuple sequences.
    out = []
    for inst in results:
        conv = []
        for clique in inst:
            conv.append([int(v) for v in clique])
        out.append(conv)
    return out
