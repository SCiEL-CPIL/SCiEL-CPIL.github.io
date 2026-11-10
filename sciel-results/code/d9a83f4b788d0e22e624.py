"""Clique partition of split graphs with ascending-degree greedy and local search."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build cross-edge adjacency: N[i] = set of C-vertices adjacent to I-vertex i
        N = {i: set() for i in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)

        # C-edges (all pairs within C)
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))

        I_vertices = list(range(k, n))

        def greedy_partition(order):
            """Greedy partition for a given order of I-vertices."""
            covered_C_edges = set()
            cliques = []

            for i in order:
                if not N[i]:
                    continue

                remaining_N = set(N[i])
                while remaining_N:
                    # Build largest possible S: add vertices in order, skip if conflict
                    S = set()
                    for u in sorted(remaining_N):
                        can_add = True
                        for v in S:
                            e = (min(u, v), max(u, v))
                            if e in covered_C_edges:
                                can_add = False
                                break
                        if can_add:
                            S.add(u)

                    if not S:
                        S = {min(remaining_N)}

                    clique = sorted([i] + list(S))
                    cliques.append(clique)

                    # Mark C-edges within S as covered
                    S_list = sorted(S)
                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            covered_C_edges.add((S_list[a], S_list[b]))

                    remaining_N -= S

            # Cover remaining C-edges
            remaining_C_edges = C_edges - covered_C_edges
            if remaining_C_edges:
                # Greedy: repeatedly pick the edge with most common neighbors
                rem_adj = defaultdict(set)
                for u, v in remaining_C_edges:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)

                while remaining_C_edges:
                    best_edge = None
                    best_common = -1
                    for u, v in remaining_C_edges:
                        common = rem_adj[u] & rem_adj[v]
                        if len(common) > best_common:
                            best_common = len(common)
                            best_edge = (u, v)

                    if best_edge is None:
                        u, v = next(iter(remaining_C_edges))
                        S = {u, v}
                    else:
                        u, v = best_edge
                        S = {u, v} | (rem_adj[u] & rem_adj[v])

                    S_list = sorted(S)
                    cliques.append(S_list)

                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            remaining_C_edges.discard((S_list[a], S_list[b]))
                    for node in S_list:
                        rem_adj.pop(node, None)
                        for w in list(rem_adj.keys()):
                            rem_adj[w].discard(node)
                    rem_adj = {u2: v2 for u2, v2 in rem_adj.items() if v2}

            return cliques, covered_C_edges

        # Generate initial solution with ascending degree order (small-neighborhood first)
        order_asc = sorted(I_vertices, key=lambda i: len(N[i]))
        cliques, covered_C_edges = greedy_partition(order_asc)

        # Also try descending degree order
        order_desc = sorted(I_vertices, key=lambda i: len(N[i]), reverse=True)
        cliques_desc, _ = greedy_partition(order_desc)

        # Pick the better of the two
        if len(cliques_desc) < len(cliques):
            cliques = cliques_desc

        best_cliques = cliques
        best_count = len(cliques)

        # Try multiple random orderings
        rng = random.Random(seed + 1000)
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            rand_order = rng.sample(I_vertices, len(I_vertices))
            c, _ = greedy_partition(rand_order)
            if len(c) < best_count:
                best_count = len(c)
                best_cliques = c

        # Local search: try to expand cross-cliques to cover more C-edges
        # and then remove redundant C-only cliques
        def get_coverage(cliques_list):
            """Return (covered_by_cross, covered_by_conly, cross_cliques, conly_cliques)"""
            covered_by_cross = set()
            covered_by_conly = set()
            cross_cliques = []
            conly_cliques = []

            for clique in cliques_list:
                c_verts = [v for v in clique if v < k]
                i_verts = [v for v in clique if v >= k]

                if len(i_verts) == 1 and len(c_verts) >= 1:
                    i = i_verts[0]
                    S = set(c_verts)
                    cross_cliques.append((i, S))
                    S_list = sorted(S)
                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            covered_by_cross.add((S_list[a], S_list[b]))
                elif len(i_verts) == 0 and len(c_verts) >= 2:
                    # C-only clique
                    conly_cliques.append(set(c_verts))
                    c_list = sorted(c_verts)
                    for a in range(len(c_list)):
                        for b in range(a + 1, len(c_list)):
                            covered_by_conly.add((c_list[a], c_list[b]))

            return covered_by_cross, covered_by_conly, cross_cliques, conly_cliques

        # Local search: expand cross-cliques to absorb C-only clique edges
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            covered_by_cross, covered_by_conly, cross_cliques, conly_cliques = get_coverage(best_cliques)

            # Try expanding each cross-clique
            new_cross_cliques = []
            for i, S in cross_cliques:
                expanded = set(S)
                # Try to add vertices from N(i) \ S
                candidates = list(N[i] - S)
                for w in candidates:
                    # Check if we can add w: for all v in expanded, edge (w,v) not in covered_by_cross
                    can_add = True
                    for v in expanded:
                        e = (min(w, v), max(w, v))
                        if e in covered_by_cross:
                            can_add = False
                            break
                    if can_add:
                        expanded.add(w)
                new_cross_cliques.append((i, expanded))

            # Rebuild cliques from new_cross_cliques and conly_cliques
            new_cliques = []
            new_covered_by_cross = set()
            for i, S in new_cross_cliques:
                if S:
                    clique = sorted([i] + list(S))
                    new_cliques.append(clique)
                    S_list = sorted(S)
                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            new_covered_by_cross.add((S_list[a], S_list[b]))

            # Add C-only cliques only if they still have uncovered edges
            for S in conly_cliques:
                S_list = sorted(S)
                has_uncovered = False
                for a in range(len(S_list)):
                    for b in range(a + 1, len(S_list)):
                        e = (S_list[a], S_list[b])
                        if e not in new_covered_by_cross:
                            has_uncovered = True
                            break
                    if has_uncovered:
                        break
                if has_uncovered:
                    new_cliques.append(S_list)

            if len(new_cliques) < best_count:
                best_count = len(new_cliques)
                best_cliques = new_cliques
                improved = True

        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        # Ensure all are Python ints and lists
        results.append([[int(v) for v in clique] for clique in cliques])

    return results
# EVOLVE-BLOCK-END
