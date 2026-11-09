"""Rewritten program using greedy clique partitioning with backtracking fallback."""
import numpy as np
from itertools import combinations


def construct():
    def instance(n, p, seed):
        rng = np.random.default_rng(seed)
        # Generate edge list in the same order as the original random.Random approach
        # We need to replicate: for u in range(n): for v in range(u+1, n): if rng.random() < p
        # np.random.default_rng gives different sequence, so use Python's random instead
        import random
        r = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if r.random() < p:
                    edges.append((u, v))
        return edges

    def greedy_clique_partition(edges, n):
        # Build adjacency as bitmasks for fast clique checks
        adj = [0] * n
        edge_set = set(edges)
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u

        # Remaining edges to assign
        remaining = set(edges)

        # Sort edges by "clique potential" - edges whose endpoints have many common neighbors first
        # Precompute common neighbor counts for each edge
        def common_neighbors(u, v):
            return adj[u] & adj[v] & ~(1 << u) & ~(1 << v)

        edge_potential = {}
        for e in edges:
            u, v = e
            cn = common_neighbors(u, v)
            edge_potential[e] = bin(cn).count('1')

        # Sort edges descending by potential
        sorted_edges = sorted(edges, key=lambda e: -edge_potential[e])

        cliques = []

        # For each edge in order, try to extend it into the largest possible clique
        # using only remaining edges
        for e in sorted_edges:
            if e not in remaining:
                continue
            u, v = e
            # Find all common neighbors that form a clique with u,v using only remaining edges
            # Start with {u, v} and greedily add vertices
            current = [u, v]
            current_set = {u, v}
            # Candidate vertices: common neighbors of u and v
            cn = common_neighbors(u, v)
            candidates = []
            for w in range(n):
                if cn & (1 << w):
                    candidates.append(w)

            # Sort candidates by how many remaining edges they share with current clique
            def remaining_edges_to_clique(w, clique):
                cnt = 0
                for x in clique:
                    if (min(w, x), max(w, x)) in remaining:
                        cnt += 1
                return cnt

            # Greedily add vertices
            added = True
            while added:
                added = False
                # Recompute candidates that still connect to all current clique members via remaining edges
                valid_candidates = []
                for w in candidates:
                    if w in current_set:
                        continue
                    ok = True
                    for x in current:
                        edge = (min(w, x), max(w, x))
                        if edge not in remaining:
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(w)
                if not valid_candidates:
                    break
                # Pick the one that maximizes remaining edges to current clique
                best_w = max(valid_candidates, key=lambda w: remaining_edges_to_clique(w, current))
                current.append(best_w)
                current_set.add(best_w)
                added = True

            # Now remove all edges within this clique from remaining
            clique_edges = set()
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    a, b = current[i], current[j]
                    edge = (min(a, b), max(a, b))
                    if edge in remaining:
                        clique_edges.add(edge)

            if len(current) >= 2 and clique_edges:
                remaining -= clique_edges
                cliques.append(current)
            else:
                # Shouldn't happen since we started with a remaining edge
                remaining.discard(e)
                cliques.append([u, v])

        # Any leftover edges become 2-cliques
        for e in remaining:
            cliques.append(list(e))

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = greedy_clique_partition(edges, n)
        result.append(cliques)

    return result
