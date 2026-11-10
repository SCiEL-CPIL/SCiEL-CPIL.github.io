"""Clique partition of split graphs: pair-packing + greedy-coloring hybrid."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        I = list(range(k, n))
        C = list(range(k))

        # Build cross-neighborhoods
        cross = {i: set() for i in I}
        for (a, b) in edges:
            if a < k and b >= k:
                cross[b].add(a)
            elif b < k and a >= k:
                cross[a].add(b)

        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))
        used_C_edges = set()
        mixed_cliques = []

        # Process I-vertices in ascending degree
        for i in sorted(I, key=lambda x: len(cross[x])):
            neighbors = cross[i]
            if not neighbors:
                continue

            # Build conflict graph on neighbors:
            # edge (u,w) if (u,w) is a C-edge already used
            # We want to partition neighbors into independent sets of this graph
            # Each independent set S gives clique {i} + S
            # But we also want to CONSUME C-edges: if u,w in S and (u,w) unused,
            # we add it to used_C_edges.
            #
            # Strategy: greedily build cliques. Start with a vertex, add compatible
            # vertices (no conflict = no used C-edge between them). Prefer vertices
            # that form unused C-edges with current clique (to consume them).

            remaining = set(neighbors)
            local_used = set()  # C-edges consumed by this i-vertex's cliques

            while remaining:
                if time.time() > deadline:
                    break
                # Pick a starting vertex: highest degree in remaining (in C-graph)
                # i.e., most unused C-edges to other remaining vertices
                best_start = None
                best_deg = -1
                for v in remaining:
                    deg = sum(1 for w in remaining if w != v and frozenset((v, w)) in all_C_edges and frozenset((v, w)) not in used_C_edges and frozenset((v, w)) not in local_used)
                    if deg > best_deg:
                        best_deg = deg
                        best_start = v

                if best_start is None:
                    break

                # Greedily build a clique: start with best_start
                clique = [best_start]
                candidates = set(remaining - {best_start})

                while candidates:
                    # Find a vertex that can join: no used C-edge with any in clique
                    # Prefer one that maximizes unused C-edges to clique (to consume)
                    best_cand = None
                    best_score = -1
                    for v in candidates:
                        # Check compatibility: no used_C_edges or local_used edge with any clique member
                        compatible = True
                        consume_count = 0
                        for u in clique:
                            e = frozenset((v, u))
                            if e in used_C_edges or e in local_used:
                                compatible = False
                                break
                            if e in all_C_edges:
                                consume_count += 1
                        if compatible and consume_count > best_score:
                            best_score = consume_count
                            best_cand = v

                    if best_cand is None:
                        # No vertex with positive consume; try any compatible vertex
                        for v in candidates:
                            compatible = True
                            for u in clique:
                                e = frozenset((v, u))
                                if e in used_C_edges or e in local_used:
                                    compatible = False
                                    break
                            if compatible:
                                best_cand = v
                                break

                    if best_cand is None:
                        break

                    clique.append(best_cand)
                    candidates.discard(best_cand)
                    # Remove candidates that conflict with best_cand
                    new_candidates = set()
                    for v in candidates:
                        e = frozenset((v, best_cand))
                        if e not in used_C_edges and e not in local_used:
                            new_candidates.add(v)
                    candidates = new_candidates

                # Form the clique {i} + clique
                if len(clique) >= 1:
                    mixed_cliques.append([i] + clique)
                    # Consume all C-edges within clique
                    for idx, u in enumerate(clique):
                        for w in clique[idx + 1:]:
                            e = frozenset((u, w))
                            if e in all_C_edges and e not in used_C_edges:
                                local_used.add(e)
                                used_C_edges.add(e)

                remaining.difference_update(clique)

        # Cover remaining C-edges with pure-C cliques
        remaining_C_edges = all_C_edges - used_C_edges
        pure_cliques = []
        remaining = set(remaining_C_edges)

        while remaining:
            if time.time() > deadline:
                break
            # Find a maximal clique in the graph of remaining edges
            # Build adjacency from remaining edges
            verts_in_rem = set()
            for e in remaining:
                verts_in_rem.update(e)

            if not verts_in_rem:
                break

            # Compute degrees
            degs = {v: 0 for v in verts_in_rem}
            for e in remaining:
                degs[list(e)[0]] += 1
                degs[list(e)[1]] += 1

            # Pick vertex with max degree
            start_v = max(verts_in_rem, key=lambda x: degs[x])

            # Greedy maximal clique
            clique = {start_v}
            candidates = set(verts_in_rem - {start_v})

            # Filter candidates to neighbors of start_v in remaining
            candidates = {v for v in candidates if frozenset((start_v, v)) in remaining}

            while candidates:
                # Pick candidate with max edges to current clique
                best_cand = None
                best_score = -1
                for v in candidates:
                    score = sum(1 for u in clique if frozenset((v, u)) in remaining)
                    if score == len(clique) and score > best_score:
                        best_score = score
                        best_cand = v
                if best_cand is None:
                    # Try any that connects to all
                    for v in candidates:
                        score = sum(1 for u in clique if frozenset((v, u)) in remaining)
                        if score == len(clique):
                            best_cand = v
                            break
                if best_cand is None:
                    break
                clique.add(best_cand)
                candidates.discard(best_cand)
                # Filter: must connect to all clique members
                new_candidates = set()
                for v in candidates:
                    ok = True
                    for u in clique:
                        if u != v and frozenset((v, u)) not in remaining:
                            ok = False
                            break
                    if ok:
                        new_candidates.add(v)
                candidates = new_candidates

            # Remove all edges within clique
            to_remove = set()
            clique_list = list(clique)
            for idx, u in enumerate(clique_list):
                for w in clique_list[idx + 1:]:
                    e = frozenset((u, w))
                    if e in remaining:
                        to_remove.add(e)
            remaining -= to_remove

            if len(clique) >= 2:
                pure_cliques.append(sorted(clique))

        result = mixed_cliques + pure_cliques
        return [list(c) for c in result]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
# EVOLVE-BLOCK-END
