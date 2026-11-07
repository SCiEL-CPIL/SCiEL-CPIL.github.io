"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = 150.0
    SAFETY = 15.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_split(n, k, p, seed):
        rng = Random(seed)
        # big clique: 0..k-1 ; independent: k..n-1
        neigh = {}  # v -> frozenset of clique neighbors
        for v in range(k, n):
            s = set()
            for u in range(k):
                if rng.random() < p:
                    s.add(u)
            neigh[v] = frozenset(s)
        return neigh

    def solve_one(neigh, k):
        # Vertices: 0..k-1 clique (complete), k..n-1 independent.
        # Every clique may contain at most one independent vertex.
        # We must cover every edge exactly once with cliques (each size >= 2).
        #
        # Strategy:
        #  - The big clique edges (u1,u2) must be covered. We can use the full
        #    clique {0..k-1} to cover ALL clique-clique edges at once (one clique).
        #    Then no other clique may contain two clique vertices, else it'd
        #    re-cover a clique-clique edge. So every other clique = {u, v} with
        #    u clique, v independent, OR {v} alone is not allowed (size>=2).
        #    Actually a clique of a single independent vertex + one clique vertex
        #    = edge (u,v), size 2, valid.
        #  - Each independent vertex v has edges (u,v) for u in N(v). These must
        #    each be in exactly one clique. Since v can't be with another
        #    independent vertex, each edge (u,v) must be its own clique {u,v}
        #    (a 2-clique), OR grouped? {u1, u2, v} is a clique (u1-u2 edge exists,
        #    u1-v, u2-v exist) but it would cover edge (u1,u2) which is already
        #    covered by the big clique -> violation.
        #    So we CANNOT put two clique vertices with an independent vertex.
        #  - Therefore each cross edge (u,v) must be its own 2-clique {u,v}.
        #
        #  Total = 1 (big clique) + (number of cross edges).
        cliques = [list(range(k))]
        for v in sorted(neigh.keys()):
            for u in sorted(neigh[v]):
                cliques.append([u, v])
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        if time.time() > DEADLINE - SAFETY:
            break
        n, k, p, seed = spec
        neigh = build_split(n, k, p, seed)
        result.append(solve_one(neigh, k))

    while len(result) < 3:
        result.append([])
    return result
# EVOLVE-BLOCK-END
