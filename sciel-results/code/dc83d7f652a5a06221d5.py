"""Clique partition of split graphs via biclique packing of the cross graph."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed, deadline):
        rng = Random(seed)
        cross = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            if nbrs:
                cross[v] = nbrs
        if not cross:
            return [[list(range(k))]]
        # Greedy biclique packing of the bipartite cross graph.
        # Each biclique C x I becomes one clique C union I covering all its edges.
        # Greedy heuristic: keep the densest complete sub-rectangle.
        # Work with sets for speed.
        left_set = {u: set(cross[v] for v in cross) }  # not needed; use direct
        # Maintain for each independent vertex its current neighbor set (left vertices).
        # We'll pick a biclique by choosing a left vertex set C and right vertex set I
        # such that every v in I has all of C in its neighbors, maximizing |C|*|I|.
        # Efficient approach: repeatedly pick the right vertex with most neighbors,
        # take C = its neighbor set, then I = right vertices containing C.
        # Then recurse on the remainder.
        # Represent right side as dict v -> frozenset of left neighbors (mutable via list).
        right = {v: set(nbrs) for v, nbrs in cross.items()}
        cliques = []
        # Full clique-part clique covers all internal clique-part edges.
        cliques.append(list(range(k)))
        # Greedy biclique extraction.
        while right:
            if time.time() > deadline:
                # Fallback: cover each remaining edge as a 2-clique.
                for v, nbrs in right.items():
                    for u in nbrs:
                        cliques.append([u, v])
                return cliques
            # Pick right vertex with largest neighbor set.
            v0 = max(right, key=lambda v: len(right[v]))
            C = set(right[v0])
            # I = all right vertices whose neighbor set is a superset of C.
            I = [v for v in right if C.issubset(right[v])]
            # Clique: C union I
            clique = sorted(list(C) + I)
            cliques.append(clique)
            # Remove covered edges: for each v in I, remove C from its neighbor set.
            for v in I:
                right[v] -= C
            # Drop right vertices with no remaining neighbors.
            for v in list(right):
                if not right[v]:
                    del right[v]
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    out = []
    for spec in specs:
        n, k, p, seed = spec
        part = build_partition(n, k, p, seed, deadline)
        # Ensure each clique is a list of distinct ints, at least two vertices.
        clean = []
        for c in part:
            c = list(c)
            if len(c) >= 2:
                clean.append(c)
        out.append(clean)
    return out
# EVOLVE-BLOCK-END
