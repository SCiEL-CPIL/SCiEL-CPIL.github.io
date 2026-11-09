"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []

        # Build adjacency bitmask for each I vertex over C vertices
        # adj_mask[v] is an integer where bit u is set iff (u,v) is an edge
        adj_mask = {}
        for v in range(k, n):
            mask = 0
            for u in range(k):
                if (u, v) in edges:
                    mask |= (1 << u)
            adj_mask[v] = mask

        # Group I vertices by identical adjacency mask
        # Vertices with the same mask form a clique together with their common neighbors in C
        from collections import defaultdict
        mask_groups = defaultdict(list)
        for v in range(k, n):
            mask_groups[adj_mask[v]].append(v)

        # Track covered C-C edges
        covered_cc = set()

        # Process groups: for each mask, the vertices in that group plus all C-neighbors form a clique
        # But we need to be careful: the C-neighbors' edges among themselves must not be double-covered
        # Strategy: for each group, if the group's C-neighborhood has size >= 1, form clique = group + C-neighbors
        # This covers all cross edges for those I vertices AND some C-C edges within the neighborhood

        # Sort groups by mask popcount descending (larger neighborhoods first to cover more C-C edges)
        groups = sorted(mask_groups.items(), key=lambda x: bin(x[0]).count('1'), reverse=True)

        for mask, group in groups:
            if mask == 0:
                # No cross edges for these vertices; they can't be in any clique with C vertices
                # They have no edges to cover, skip
                continue
            if len(group) < 1:
                continue
            # The clique is group + all C vertices in the mask
            c_verts = [u for u in range(k) if (mask >> u) & 1]
            if len(c_verts) + len(group) < 2:
                continue
            clique = sorted(group + c_verts)
            # Verify all pairs in clique are edges (they should be by construction for split graph)
            # Cross edges: guaranteed by mask
            # C-C edges: guaranteed since C is a clique
            # I-I edges: none needed since I is independent, but we only need each edge covered exactly once
            # The I-I "edges" don't exist, so no issue
            cliques.append(clique)
            # Mark C-C edges within c_verts as covered
            for i in range(len(c_verts)):
                for j in range(i + 1, len(c_verts)):
                    covered_cc.add((c_verts[i], c_verts[j]))

        # Now cover remaining C-C edges
        remaining_edges = set()
        for u in range(k):
            for u2 in range(u + 1, k):
                if (u, u2) in edges and (u, u2) not in covered_cc:
                    remaining_edges.add((u, u2))

        if remaining_edges:
            # Build adjacency for remaining C-C graph
            rem_adj = {u: set() for u in range(k)}
            for (u, v) in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)

            remaining_vertices = set(rem_adj.keys())

            # DSATUR-style maximum clique cover: repeatedly find a maximum clique
            while remaining_vertices and remaining_edges:
                if time.time() > DEADLINE:
                    break
                # Pick starting vertex with highest degree in remaining graph
                start = max(remaining_vertices, key=lambda x: len(rem_adj[x]))
                S = {start}
                candidates = set(remaining_vertices - S)

                # Greedy grow: repeatedly add vertex with max connections to S
                while True:
                    best = None
                    best_score = -1
                    for u in candidates:
                        score = len(rem_adj[u] & S)
                        if score > best_score:
                            best_score = score
                            best = u
                    if best is None or best_score < len(S):
                        break
                    S.add(best)
                    candidates.discard(best)

                if len(S) >= 2:
                    clique = sorted(S)
                    cliques.append(clique)
                    # Remove edges within S
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            e = (clique[i], clique[j])
                            if e in remaining_edges:
                                remaining_edges.discard(e)
                    for u in S:
                        remaining_vertices.discard(u)
                        for v in S:
                            if u != v:
                                rem_adj[u].discard(v)
                                rem_adj[v].discard(u)
                else:
                    # Single vertex with no edges? Shouldn't happen if remaining_edges non-empty
                    remaining_vertices.discard(start)

            # Fallback: cover any leftover edges as 2-cliques
            for e in remaining_edges:
                cliques.append(list(e))

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
# EVOLVE-BLOCK-END
