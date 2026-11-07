"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition(n, k, p, seed):
        # Clique part is vertices 0..k-1, independent set is k..n-1.
        # All edges within the clique part must lie in exactly one clique,
        # and since the clique part is a clique, the unique clique that can
        # contain every such edge without violating "exactly one" is the
        # whole clique part itself.
        cliques = [list(range(k))]
        # Each independent-set vertex v can be combined with the entire
        # clique part to form a clique (since every vertex in the clique
        # part is adjacent to every other, and v is adjacent to a subset).
        # The clique {0..k-1, v} covers exactly the cross edges incident to v,
        # and no other cross edge. This is minimal: two independent-set
        # vertices cannot share a clique (they are non-adjacent), and every
        # cross edge must be covered.
        for v in range(k, n):
            cliques.append(list(range(k)) + [v])
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = partition(n, k, p, seed)
        # Ensure each clique is a Python list of Python ints, length >= 2.
        cleaned = []
        for c in cliques:
            cleaned.append([int(x) for x in c])
        result.append(cleaned)
    return result
# EVOLVE-BLOCK-END
