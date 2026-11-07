import time
from random import Random
from collections import deque
from itertools import combinations

def construct():
    _T0 = time.time()
    _DEADLINE = 130.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Build adjacency
        adj = [set() for _ in range(n)]
        for (a, b) in edge_set:
            adj[a].add(b)
            adj[b].add(a)

        # N_C[v] = set of C-vertices adjacent to I-vertex v
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if u in adj[v])

        # All C-C edges
        all_cc = set((u, w) for u in range(k) for w in range(u + 1, k))

        # Strategy: Group I-vertices into cliques.
        # A clique containing I-vertices S and C-vertices T must satisfy:
        #   - Every pair in T is an edge (always true, C is complete)
        #   - For every v in S and every u in T: (v,u) is an edge, i.e., u in N_C[v]
        #   - So T must be subset of intersection of N_C[v] for all v in S
        #   - Also, no two vertices in S can be adjacent (I is independent) - always true
        #   - The clique covers: all edges within T (C-C), and all edges between S and T (cross)
        #
        # We want to partition all cross edges and C-C edges into minimum cliques.
        #
        # Approach: Greedy set cover on uncovered edges.
        # At each step, find the clique (subset of vertices) that covers the most uncovered edges.
        # Restrict to cliques that are "maximal" in some sense.
        #
        # For efficiency, we consider:
        # 1. Clauses with I-vertices: pick a subset S of I-vertices, T = intersection of N_C[v] for v in S
        #    The clique is S ∪ T. It covers cross edges from S to T and C-C edges within T.
        # 2. C-only cliques: any subset of C vertices.

        uncovered = set(edge_set)
        cliques = []

        # Precompute: for each I-vertex, its N_C
        # For grouping: try pairs and small groups of I-vertices

        # Phase 1: Greedily cover cross edges using I-vertex cliques
        # At each step, find the best clique involving at least one I-vertex
        # that covers the most uncovered edges.

        i_vertices = list(range(k, n))
        # Track which I-vertices still have uncovered cross edges
        i_active = set(v for v in i_vertices if any((v, u) in uncovered or (u, v) in uncovered for u in range(k)))

        while i_active and not _time_up():
            best_clique = None
            best_count = 0

            # Try single I-vertices
            for v in list(i_active):
                nc = N_C[v]
                if not nc:
                    continue
                # Clique: {v} ∪ nc
                # Covers: cross edges (v,u) for u in nc, and C-C edges within nc
                count = 0
                for u in nc:
                    if (min(v,u), max(v,u)) in uncovered:
                        count += 1
                nc_list = sorted(nc)
                for i in range(len(nc_list)):
                    for j in range(i+1, len(nc_list)):
                        e = (nc_list[i], nc_list[j])
                        if e in uncovered:
                            count += 1
                if count > best_count:
                    best_count = count
                    best_clique = sorted([v] + nc_list)

            # Try pairs of I-vertices (if not too many)
            if len(i_active) <= 50:
                for v1, v2 in combinations(sorted(i_active), 2):
                    nc1 = N_C[v1]
                    nc2 = N_C[v2]
                    common = nc1 & nc2
                    if len(common) < 2:
                        continue
                    # Clique: {v1, v2} ∪ common
                    count = 0
                    # Cross edges from v1 to common
                    for u in common:
                        if (min(v1,u), max(v1,u)) in uncovered:
                            count += 1
                    # Cross edges from v2 to common
                    for u in common:
                        if (min(v2,u), max(v2,u)) in uncovered:
                            count += 1
                    # C-C edges within common
                    common_list = sorted(common)
                    for i in range(len(common_list)):
                        for j in range(i+1, len(common_list)):
                            e = (common_list[i], common_list[j])
                            if e in uncovered:
                                count += 1
                    if count > best_count:
                        best_count = count
                        best_clique = sorted([v1, v2] + common_list)

            if best_clique is None or best_count == 0:
                break

            cliques.append(best_clique)
            # Remove covered edges
            cl_set = set(best_clique)
            for e in list(uncovered):
                if e[0] in cl_set and e[1] in cl_set:
                    uncovered.discard(e)

            # Update i_active
            new_i_active = set()
            for v in i_active:
                # Check if v still has uncovered cross edges
                has_edge = False
                for u in range(k):
                    e = (min(v,u), max(v,u))
                    if e in uncovered:
                        has_edge = True
                        break
                if has_edge:
                    new_i_active.add(v)
            i_active = new_i_active

        # Phase 2: Cover remaining C-C edges with C-only cliques
        remaining_cc = set(e for e in uncovered if e[0] < k and e[1] < k)
        if remaining_cc:
            remaining_set = remaining_cc
            c_cliques = []
            while remaining_set and not _time_up():
                # Find C-vertex with most remaining incident edges
                deg = [0] * k
                for (u, w) in remaining_set:
                    deg[u] += 1
                    deg[w] += 1

                best_v = max(range(k), key=lambda x: deg[x])
                if deg[best_v] == 0:
                    break

                # Build largest clique around best_v covering most remaining edges
                # Start with {best_v}, add all C-vertices that have remaining edges to best_v
                # Actually, in complete C, we can add any C-vertex.
                # Greedy: add vertices in order of how many remaining edges they cover
                clique = [best_v]
                candidates = [v for v in range(k) if v != best_v]
                # Sort by number of remaining edges incident to v
                cand_deg = [0] * k
                for (u, w) in remaining_set:
                    cand_deg[u] += 1
                    cand_deg[w] += 1
                candidates.sort(key=lambda x: -cand_deg[x])

                for v in candidates:
                    if _time_up():
                        break
                    # Check if adding v covers at least one remaining edge
                    covers = False
                    for u in clique:
                        e = (min(v,u), max(v,u))
                        if e in remaining_set:
                            covers = True
                            break
                    if covers:
                        clique.append(v)

                if len(clique) < 2:
                    # Try just best_v with its highest-degree neighbor
                    best_neighbor = max(range(k), key=lambda x: cand_deg[x]) if cand_deg else best_v
                    if best_neighbor != best_v:
                        clique = [best_v, best_neighbor]
                    else:
                        break

                clique_set = set(clique)
                for e in list(remaining_set):
                    if e[0] in clique_set and e[1] in clique_set:
                        remaining_set.discard(e)

                c_cliques.append(sorted(clique))

            cliques.extend(c_cliques)

        # Handle any remaining edges (shouldn't happen, but safety)
        if uncovered:
            for e in uncovered:
                cliques.append(sorted(e))

        # Verify validity
        if not cliques:
            for (a, b) in edge_set:
                cliques.append(sorted([a, b]))
            return cliques

        for cl in cliques:
            if len(cl) < 2:
                cliques = []
                for (a, b) in edge_set:
                    cliques.append(sorted([a, b]))
                return cliques
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        cliques = []
                        for (a2, b2) in edge_set:
                            cliques.append(sorted([a2, b2]))
                        return cliques

        # Verify all edges covered exactly once
        edge_cover_count = {}
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    edge_cover_count[(a, b)] = edge_cover_count.get((a, b), 0) + 1

        for e in edge_set:
            if edge_cover_count.get(e, 0) != 1:
                cliques = []
                for (a, b) in edge_set:
                    cliques.append(sorted([a, b]))
                return cliques

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append(sorted([u, v]))
            results.append(cliques)
            continue
        n, k, p, seed = spec
        cliques = _solve_instance(n, k, p, seed)
        results.append(cliques)
    return results
