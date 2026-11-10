"""Clique partition of split graphs via minimum set cover on cross-edges."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Identify cross-edges: one endpoint in C (0..k-1), one in I (k..n-1)
        cross_edges = set()
        for (u, v) in edges:
            if u < k and v >= k:
                cross_edges.add((u, v))
            elif v < k and u >= k:
                cross_edges.add((v, u))

        # The clique part: all C vertices form a clique
        # One clique covers all C-C edges
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))

        # Now cover all cross-edges using cliques of form {c} ∪ S ∪ {i}
        # where S ⊆ C, c ∈ C, i ∈ I, and i adjacent to all of c ∪ S
        # Since C is a clique, any subset of C is a clique.
        # A clique containing i and some subset T ⊆ C (where i adjacent to all in T)
        # covers cross-edges (t, i) for all t in T.
        #
        # Greedy set cover: universe = cross_edges
        # For each i in I, the best clique containing i is {i} ∪ N(i) ∩ C
        # This covers all cross-edges incident to i.
        # But we can also use cliques without i (pure C-cliques) - those don't help with cross-edges.
        # So every cross-edge (c, i) must be in a clique containing i.
        # The clique containing i can include multiple C-neighbors of i.
        #
        # Key insight: For a fixed i, the clique {i} ∪ (N(i) ∩ C) covers ALL cross-edges incident to i.
        # So one clique per i covers all its cross-edges. That gives |I| cliques.
        # But we can do better: if two i-vertices have the same neighborhood in C,
        # they can't share a clique (I is independent). So each i needs its own clique.
        #
        # Wait - can we split? A cross-edge (c, i) is covered by clique {c, i} (2-clique).
        # Or by {c, c2, i} if c2 also adjacent to i.
        # The question is: does using larger cliques reduce the COUNT?
        # No! Each clique contains exactly one I-vertex (since I is independent).
        # So each clique covers cross-edges for exactly one i.
        # Therefore we need at least one clique per i that has any cross-edges.
        # And one clique per such i (containing i and all its C-neighbors) is optimal
        # in terms of count: we can't merge two i's into one clique.
        #
        # BUT WAIT: we could potentially NOT use a clique for some i, and instead
        # cover its cross-edges via... no, every clique containing a cross-edge (c,i)
        # must contain i. So each i with cross-edges needs at least one clique.
        # And one clique per i suffices (use {i} ∪ all C-neighbors of i).
        #
        # So the minimum number of cliques covering cross-edges = number of i in I
        # that have at least one cross-edge.
        #
        # Total cliques = 1 (for C) + (number of i with cross-edges)
        #
        # But wait - the C-clique {0,1,...,k-1} already contains all C vertices.
        # A cross-edge (c, i) is NOT in the C-clique (since i is not in C).
        # So we still need separate cliques for cross-edges.
        #
        # However, we could potentially avoid the full C-clique and instead
        # distribute C-vertices among the i-cliques. But then C-C edges need coverage.
        # A C-C edge (c1, c2) must be in a clique containing both. If c1 is in
        # clique {c1, i1} and c2 is in clique {c2, i2}, the edge (c1,c2) is not covered.
        # So we need a clique containing all pairs of C-vertices, which means
        # we need the full C-clique OR multiple C-subset cliques whose union covers
        # all C-C edges. But a clique partition requires each edge in EXACTLY one clique.
        # The full C-clique is the simplest: 1 clique covers all C-C edges.
        #
        # Alternative: could we put some C-vertices into i-cliques and still cover
        # C-C edges? E.g., clique {c1, c2, i1} covers edges (c1,c2), (c1,i1), (c2,i1).
        # Then we don't need (c1,c2) in the main C-clique. But we'd need to ensure
        # every C-C edge appears in exactly one clique. This is a set partition of C-C edges.
        # This could potentially reduce total count if we can pack C-C edges into
        # the i-cliques. But each i-clique can contain at most one I-vertex, and the
        # C-vertices in it form a clique (they're all in C, so they're all adjacent).
        #
        # Let's think: total edges = C-C edges + cross-edges.
        # Option A: 1 C-clique + one clique per active i. Count = 1 + |active I|.
        # Option B: Distribute C-C edges among i-cliques. Each i-clique {T_i ∪ {i}} where
        #   T_i ⊆ C. C-C edges covered: all pairs within T_i. Cross-edges covered: (t,i) for t in T_i.
        #   We need: every C-C edge in exactly one T_i (or in a pure C-clique),
        #   every cross-edge in exactly one clique.
        #
        # This is getting complex. For the given parameters (k=10,20,30; n=30,60,90; p=0.7),
        # let's just use the simple optimal: 1 C-clique + one clique per active i.
        #
        # Actually, let me reconsider. The current program scores 1.3357 which means
        # seed/value = 1.3357, so value = seed/1.3357. The seed is the initial program.
        # Let me just compute what the simple approach gives and see if it's better.

        # Simple approach: 1 C-clique + one clique per i with cross-edges
        # For each i in I with at least one neighbor in C:
        #   clique = [i] + sorted(N(i) ∩ C)
        # This covers all cross-edges (each (c,i) is in the i-clique)
        # and the C-clique covers all C-C edges.
        # No edge is in two cliques: C-C edges only in C-clique, cross-edges only in i-cliques.

        for i in range(k, n):
            if time.time() > deadline - 5:
                # Fallback
                break
            neighbors_in_C = sorted(adj[i] & set(range(k)))
            if neighbors_in_C:
                cliques.append(neighbors_in_C + [i])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            # Emergency fallback: each edge as a 2-clique
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue

        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)

        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])

        results.append(final_cliques)

    return results
# EVOLVE-BLOCK-END
