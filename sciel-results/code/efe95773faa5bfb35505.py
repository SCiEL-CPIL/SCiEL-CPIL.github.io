import time
from random import Random
from collections import deque

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _dsatur_color_bipartite(c_adj, v_adj, num_c, num_v):
        """Color bipartite graph: C vertices 0..num_c-1, V vertices num_c..num_c+num_v-1.
        Returns color list (length num_c + num_v) using DSATUR greedy."""
        total = num_c + num_v
        adj = [set() for _ in range(total)]
        for c in range(num_c):
            for v in c_adj[c]:
                adj[c].add(num_c + v)
                adj[num_c + v].add(c)
        for v in range(num_v):
            for c in v_adj[v]:
                adj[num_c + v].add(c)
                adj[c].add(num_c + v)

        colors = [-1] * total
        sat_colors = [set() for _ in range(total)]
        degree = [len(adj[i]) for i in range(total)]
        uncolored = set(range(total))

        for _ in range(total):
            best_node = None
            best_key = None
            for node in uncolored:
                key = (len(sat_colors[node]), degree[node])
                if best_key is None or key > best_key:
                    best_key = key
                    best_node = node
            uncolored.discard(best_node)

            c = 0
            while c in sat_colors[best_node]:
                c += 1
            colors[best_node] = c

            for nb in adj[best_node]:
                if nb in uncolored and c not in sat_colors[nb]:
                    sat_colors[nb].add(c)

        return colors

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # N_C[v] = set of C-vertices adjacent to I-vertex v
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        cross_edges_norm = []
        for (a, b) in edge_set:
            if a < k and b >= k:
                cross_edges_norm.append((a, b))
            elif b < k and a >= k:
                cross_edges_norm.append((b, a))

        # Regime 1 baseline: 1 C-clique + per-cross-edge 2-cliques
        base_cliques = [list(range(k))]
        for (u, v) in cross_edges_norm:
            base_cliques.append(sorted([u, v]))
        best_cliques = base_cliques
        best_count = len(base_cliques)

        # Regime 2: Select I-vertices to cover all C-C edges, then color bipartite graph
        all_cc_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        total_cc = len(all_cc_edges)

        if total_cc == 0:
            return best_cliques, best_count

        # For each I-vertex, which C-C edges can it cover?
        # Edge (u,w) covered by v iff u in N_C[v] and w in N_C[v]
        # Represent as: for each v, the set of C-vertices it's adjacent to
        # C-C edges covered by v = all pairs within N_C[v]

        # Greedy set cover: select I-vertices to cover all C-C edges
        uncovered = set(all_cc_edges)
        selected_v = []
        V_v = {v: set() for v in range(k, n)}

        # Precompute for each v: set of C-C edges it covers
        v_cover = {}
        for v in range(k, n):
            nc = sorted(N_C[v])
            if len(nc) >= 2:
                v_cover[v] = set((nc[i], nc[j]) for i in range(len(nc)) for j in range(i + 1, len(nc)))
            else:
                v_cover[v] = set()

        while uncovered and not _time_up():
            best_v = None
            best_cnt = 0
            for v in range(k, n):
                if v in selected_v:
                    continue
                cnt = len(uncovered & v_cover[v])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v is None or best_cnt == 0:
                break
            selected_v.append(best_v)
            uncovered -= v_cover[best_v]

        # Local improvement: try to remove a selected v and reassign its covered edges
        # to other selected v's
        if not _time_up():
            improved = True
            while improved and not _time_up():
                improved = False
                for idx in range(len(selected_v)):
                    v_try = selected_v[idx]
                    # Check if removing v_try still allows covering all C-C edges
                    # and all cross edges for v_try are still covered
                    # Cross edges for v_try: all u with (u,v_try) in cross_edges_norm
                    # These MUST be in V_v[v_try], so if we remove v_try, those cross edges
                    # are uncovered. So we can only remove v_try if it has no cross edges.
                    has_cross = any(v == v_try for (_, v) in cross_edges_norm)
                    if has_cross:
                        continue
                    # Try removing v_try
                    temp_selected = selected_v[:idx] + selected_v[idx+1:]
                    if not temp_selected:
                        continue
                    # Check if all C-C edges can still be covered
                    all_coverable = True
                    for (u, w) in all_cc_edges:
                        if not any(u in N_C[v2] and w in N_C[v2] for v2 in temp_selected):
                            all_coverable = False
                            break
                    if all_coverable:
                        selected_v = temp_selected
                        improved = True
                        break

        if uncovered:
            # Can't cover all C-C edges with selected I-vertices
            # Try adding more or fall back
            return best_cliques, best_count

        # Now: for each selected v, determine V_v[v] (set of C-vertices in its clique)
        # Each selected v contributes exactly 1 clique: {v} ∪ V_v[v]
        # Constraints:
        #   - For cross edge (u,v): u must be in V_v[v]
        #   - For C-C edge (u,w): exactly one selected v has both u,w in V_v[v]
        #   - V_v[v] ⊆ N_C[v]
        #
        # Strategy: 
        #   1. Initialize V_v[v] with forced cross-edge assignments
        #   2. For remaining C-C edges, assign each to exactly one selected v
        #      using a greedy that prefers v with fewest current C-vertices (to keep cliques small)
        #   3. Verify no C-C edge is double-covered

        # Initialize V_v[v] with cross-edge requirements
        for (u, v) in cross_edges_norm:
            if v in selected_v:
                V_v[v].add(u)

        # Determine which C-C edges are already covered (both endpoints in V_v[v] for same v)
        # and which need assignment
        cc_edge_to_v = {}  # (u,w) -> v that covers it
        for v in selected_v:
            vc = sorted(V_v[v])
            for i in range(len(vc)):
                for j in range(i + 1, len(vc)):
                    e = (vc[i], vc[j])
                    if e in cc_edge_to_v:
                        # Double cover - problem
                        return best_cliques, best_count
                    cc_edge_to_v[e] = v

        # For remaining C-C edges, assign to a selected v
        # Use greedy: for each edge, pick the v that can cover it with smallest |V_v|
        # to balance load
        remaining_cc = [(u, w) for (u, w) in all_cc_edges if (u, w) not in cc_edge_to_v]
        
        # Sort by number of candidate v's (most constrained first)
        def _num_candidates(e):
            u, w = e
            return sum(1 for v in selected_v if u in N_C[v] and w in N_C[v])
        
        remaining_cc.sort(key=_num_candidates)
        
        for (u, w) in remaining_cc:
            if _time_up():
                break
            # Find best v: one with u,w in N_C[v] and smallest current |V_v[v]|
            best_v = None
            best_size = float('inf')
            for v in selected_v:
                if u in N_C[v] and w in N_C[v]:
                    sz = len(V_v[v])
                    if sz < best_size:
                        best_size = sz
                        best_v = v
            if best_v is None:
                return best_cliques, best_count
            V_v[best_v].add(u)
            V_v[best_v].add(w)
            cc_edge_to_v[(u, w)] = best_v

        # Now build cliques
        cliques = []
        for v in selected_v:
            if V_v[v]:
                clique = sorted([v] + list(V_v[v]))
                cliques.append(clique)
            # If V_v[v] is empty, v has no cross edges and no C-C edges assigned,
            # so it contributes no clique (and shouldn't have been selected)

        # Verify validity
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return best_cliques, best_count

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
                return best_cliques, best_count

        if len(cliques) < best_count:
            best_cliques = cliques
            best_count = len(cliques)

        return best_cliques, best_count

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
        cliques, _ = _solve_instance(n, k, p, seed)
        results.append(cliques)
    return results
