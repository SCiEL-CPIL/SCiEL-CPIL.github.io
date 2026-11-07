"""Clique partition of split graphs using optimized partitioning strategy."""
import time
from random import Random

def construct():
    _T0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Fallback: 1 clique for C, 1 per cross edge
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        # Build adjacency for I-vertices
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # For each I-vertex, find its neighbors in C
        N_C = {}
        for v in range(k, n):
            neigh = frozenset(u for u in range(k) if (u, v) in edge_set)
            N_C[v] = neigh

        # Strategy: For each I-vertex v, partition N_C[v] into subsets.
        # Each subset S (|S|>=1) gives clique {v} ∪ S.
        # C-C edges within S are also covered.
        # Remaining C-C edges covered by C-only cliques.

        # The key decision: how to partition each N_C[v].
        # We want to minimize: (total Type B cliques) + (C-only cliques for remaining C-C edges)

        # Since K_k can be covered by 1 clique, the baseline is:
        #   1 C-only clique + sum(|N_C[v]|) Type B cliques (each S has size 1)
        #   Total = 1 + sum(d_v)

        # Alternative: group neighbors. If v has d neighbors, putting them all in one set:
        #   1 Type B clique, consumes d(d-1)/2 C-C edges
        #   Remaining C-C edges: K_k minus those d(d-1)/2 edges
        #   Need to cover remaining with C-only cliques

        # For the complement: the remaining C-C graph has all edges EXCEPT those within any S used.
        # The min clique cover of remaining C-C graph = chromatic number of its complement.
        # The complement of remaining C-C graph = graph with edges exactly where C-C edges were consumed.

        # Let's try: for each v, use one big set (all neighbors together) if d_v >= 2.
        # This minimizes Type B cliques. Then cover remaining C-C edges.

        # But wait - if multiple v's consume overlapping C-C edges, we need to be careful.
        # Each C-C edge can only be in ONE clique total.

        # Let me think differently. The C-C edges must be partitioned among:
        #   - C-only cliques
        #   - Type B cliques (each Type B clique {v}∪S consumes C-C edges within S)

        # Constraint: no C-C edge is consumed by two different Type B cliques or by a Type B and a C-only clique.

        # This is a complex optimization. Let me try a greedy/heuristic approach:
        # 1. For each v, decide partition of N_C[v]
        # 2. Track which C-C edges are "used" by Type B cliques
        # 3. Cover remaining C-C edges with C-only cliques (minimum clique cover)

        # For minimum clique cover of a graph, we can use the fact that for the complement,
        # it's a coloring problem. For small k (up to 30), we can use a simple approach.

        # Actually, let me reconsider the structure. The C-C edges form K_k.
        # If we use Type B cliques that consume some C-C edges, the remaining C-C edges
        # form a subgraph of K_k. We need to cover this subgraph with cliques.

        # The minimum number of cliques to cover a graph G is the clique cover number θ(G).
        # For a complete graph minus some edges, this can be computed.

        # Let me try a simpler approach: 
        # Option A: 1 C-only clique (all of C) + 1 Type B clique per cross edge
        #   Total = 1 + |cross_edges|
        # Option B: For each v with d_v >= 2, use 1 Type B clique with all d_v neighbors
        #   This consumes C-C edges within N_C[v]. 
        #   Remaining C-C edges must be covered.
        #   But different v's may have overlapping neighborhoods, causing conflicts.

        # The conflict issue: if v1 and v2 both use a C-C edge (u,w) in their Type B cliques,
        # that edge would be in two cliques - INVALID.

        # So we need to ensure no C-C edge is consumed by more than one Type B clique.

        # This makes it a packing problem. Let me try:
        # For each v, the C-C edges consumed are E(N_C[v]) (all pairs within N_C[v]).
        # We need to select a subset of v's to use "big" Type B cliques such that
        # no C-C edge is in E(N_C[v1]) ∩ E(N_C[v2]) for selected v1, v2.

        # This is getting complex. Let me try a different approach:
        # Use the fact that for the C-C edges, the optimal cover is 1 clique (the whole C).
        # The question is whether we can save Type B cliques by "absorbing" some C-C edges.

        # Each Type B clique {v}∪S saves (|S|-1) Type B cliques compared to using |S| separate
        # {v}∪{u} cliques, but it consumes C-C edges within S.

        # The C-C edges consumed must not conflict with other Type B cliques or the C-only clique.

        # Let me try: for each v, use one Type B clique with all neighbors (if d_v >= 2).
        # Check for conflicts. If conflicts exist, fall back to individual edges.

        # Actually, the simplest improvement: if d_v = 1, we must use {v, u} (1 clique, 1 cross edge).
        # If d_v >= 2, we can use {v} ∪ N_C[v] (1 clique, d_v cross edges, d_v(d_v-1)/2 C-C edges).
        # But this C-C edge consumption might conflict.

        # Let me just try the "all in one" strategy and check validity.
        # If it's invalid, fall back to the safe strategy.

        all_cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        total_cc = len(all_cc_edges)

        def _cover_remaining_cc(used_cc, cliques_out):
            """Cover remaining C-C edges with C-only cliques via greedy coloring on complement."""
            remaining = all_cc_edges - used_cc
            if not remaining:
                return 0
            # Only consider vertices incident to remaining edges
            active = set()
            for (a, b) in remaining:
                active.add(a)
                active.add(b)
            if not active:
                return 0
            if len(remaining) == total_cc:
                cliques_out.append(list(range(k)))
                return 1
            # Check if remaining is a complete graph on active vertices
            n_active = len(active)
            if len(remaining) == n_active * (n_active - 1) // 2:
                cliques_out.append(sorted(active))
                return 1
            # Complement graph: edges where C-C edge is consumed (not in remaining)
            comp_adj = [set() for _ in range(k)]
            for (a, b) in used_cc:
                comp_adj[a].add(b)
                comp_adj[b].add(a)
            # Greedy coloring on complement restricted to active vertices
            # Color classes = cliques in remaining
            colors = {}
            uncolored = set(active)
            color_count = 0
            while uncolored:
                # Find max independent set in complement = max clique in remaining
                # Greedy: pick vertex with most neighbors in complement among uncolored
                best_v = None
                best_deg = -1
                for v in uncolored:
                    deg = len(comp_adj[v] & uncolored)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                # Build color class: greedily add vertices non-adjacent in complement
                color_class = [best_v]
                candidates = set(uncolored - {best_v})
                for v in sorted(candidates, key=lambda x: len(comp_adj[x] & candidates), reverse=True):
                    if v not in candidates:
                        continue
                    # Check v is not adjacent to any in color_class in complement
                    if all(v not in comp_adj[c] for c in color_class):
                        color_class.append(v)
                        candidates.discard(v)
                if len(color_class) >= 2:
                    cliques_out.append(list(color_class))
                    color_count += 1
                uncolored -= set(color_class)
            return color_count

        def _try_strategy(mode, order_desc=True):
            """Try a partitioning strategy. Returns (cliques, count) or None if invalid."""
            used_cc = set()
            cliques = []
            I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=order_desc)
            for v in I_verts:
                neigh = list(N_C[v])
                d = len(neigh)
                if d == 0:
                    continue
                elif d == 1:
                    cliques.append([v, neigh[0]])
                else:
                    if mode == 'big':
                        # All neighbors in one clique
                        cc_in_neigh = set()
                        for i in range(d):
                            for j in range(i + 1, d):
                                a, b = neigh[i], neigh[j]
                                e = (a, b) if a < b else (b, a)
                                cc_in_neigh.add(e)
                        if cc_in_neigh & used_cc:
                            # Conflict: fall back to individual
                            for u in neigh:
                                cliques.append([v, u])
                        else:
                            cliques.append([v] + neigh)
                            used_cc |= cc_in_neigh
                    elif mode == 'greedy_pack':
                        # Greedily pack neighbors into maximal conflict-free groups
                        remaining_neigh = list(neigh)
                        groups = []
                        while remaining_neigh:
                            group = [remaining_neigh.pop(0)]
                            changed = True
                            while changed:
                                changed = False
                                for idx in range(len(remaining_neigh)):
                                    cand = remaining_neigh[idx]
                                    ok = all(
                                        (cand, g) if cand < g else (g, cand) not in used_cc
                                        for g in group
                                    )
                                    if ok:
                                        group.append(cand)
                                        remaining_neigh.pop(idx)
                                        changed = True
                                        break
                            groups.append(group)
                        for group in groups:
                            if len(group) == 1:
                                cliques.append([v, group[0]])
                            else:
                                cliques.append([v] + group)
                                for i in range(len(group)):
                                    for j in range(i + 1, len(group)):
                                        a, b = group[i], group[j]
                                        e = (a, b) if a < b else (b, a)
                                        used_cc.add(e)
                    elif mode == 'individual':
                        for u in neigh:
                            cliques.append([v, u])
                    elif mode == 'big_then_split':
                        # Try big first; if conflict, split into two halves
                        cc_in_neigh = set()
                        for i in range(d):
                            for j in range(i + 1, d):
                                a, b = neigh[i], neigh[j]
                                e = (a, b) if a < b else (b, a)
                                cc_in_neigh.add(e)
                        if not (cc_in_neigh & used_cc):
                            cliques.append([v] + neigh)
                            used_cc |= cc_in_neigh
                        else:
                            # Try splitting into two groups
                            mid = d // 2
                            g1 = neigh[:mid]
                            g2 = neigh[mid:]
                            # Check g1
                            cc1 = set()
                            for i in range(len(g1)):
                                for j in range(i + 1, len(g1)):
                                    a, b = g1[i], g1[j]
                                    e = (a, b) if a < b else (b, a)
                                    cc1.add(e)
                            if not (cc1 & used_cc) and len(g1) >= 2:
                                cliques.append([v] + g1)
                                used_cc |= cc1
                            elif len(g1) >= 2:
                                for u in g1:
                                    cliques.append([v, u])
                            elif len(g1) == 1:
                                cliques.append([v, g1[0]])
                            # Check g2
                            cc2 = set()
                            for i in range(len(g2)):
                                for j in range(i + 1, len(g2)):
                                    a, b = g2[i], g2[j]
                                    e = (a, b) if a < b else (b, a)
                                    cc2.add(e)
                            if not (cc2 & used_cc) and len(g2) >= 2:
                                cliques.append([v] + g2)
                                used_cc |= cc2
                            elif len(g2) >= 2:
                                for u in g2:
                                    cliques.append([v, u])
                            elif len(g2) == 1:
                                cliques.append([v, g2[0]])
            # Cover remaining C-C edges
            ccount = _cover_remaining_cc(used_cc, cliques)
            return cliques, len(cliques)

        best_cliques = None
        best_count = float('inf')

        for mode in ['greedy_pack', 'big', 'big_then_split', 'individual']:
            if _time_up():
                break
            for order_desc in [True, False]:
                if _time_up():
                    break
                result = _try_strategy(mode, order_desc)
                if result is not None:
                    cliques, cnt = result
                    if cnt < best_count:
                        best_count = cnt
                        best_cliques = cliques

        # Fallback if nothing worked
        if best_cliques is None:
            best_cliques = [list(range(k))]
            for (u, v) in edge_set:
                if u >= k or v >= k:
                    best_cliques.append([u, v])
            best_count = len(best_cliques)

        results.append(best_cliques)

    return results
