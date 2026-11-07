"""Rewritten program for clique partitions of split graphs; EVOLVE-BLOCK replaced."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    deadline = time.time() + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        rng = Random(seed)
        # Build cross-neighborhoods: for each independent vertex v, which clique vertices it connects to.
        # Cross pairs are generated in order: for v in range(k, n): for u in range(k): draw.
        neigh = {}
        for v in range(k, n):
            lst = []
            for u in range(k):
                if rng.random() < p:
                    lst.append(u)
            if lst:
                neigh[v] = lst

        cliques = [[int(i) for i in range(k)]]  # the big clique C
        for v in range(k, n):
            lst = neigh.get(v)
            if lst:
                clique = [int(v)] + [int(u) for u in lst]
                cliques.append(clique)
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # safety margin: return what we have padded? No—must be valid. We have plenty of margin.
            break
        result.append(solve(*spec))

    return result
# EVOLVE-BLOCK-END
