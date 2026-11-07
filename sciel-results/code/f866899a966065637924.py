import time
from random import Random
from collections import deque
from scipy.optimize import linear_sum_assignment
import numpy as np

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

        # Baseline: 1 C-clique + per-cross-edge 2-cliques
        base_cliques = [list(range(k))]
        for (u, v) in cross_edges_norm:
            base_cliques.append(sorted([u, v]))
        best_cliques = base_cliques
        best_count = len(base_cliques)

        # All C-C edges
        all_cc_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        total_cc = len(all_cc_edges)

        if total_cc == 0:
            return best_cliques, best_count

        # STRUCTURAL CHANGE: Use set cover with local search
        # Each I-vertex v can form a clique {v} ∪ S_v where S_v ⊆ N_C[v]
        # We need to cover all C-C edges: for each (u,w), there must be exactly one v 
        # such that u,w ∈ S_v
        # Cross edges (u,v) require that if v is in selected_v, then u ∈ S_v
        
        # Precompute for each v: which C-C edges it can cover
        v_cover = {}
        for v in range(k, n):
            nc = sorted(N_C[v])
            if len(nc) >= 2:
                v_cover[v] = set((nc[i], nc[j]) for i in range(len(nc)) for j in range(i + 1, len(nc)))
            else:
                v_cover[v] = set()

        # Greedy set cover to get initial solution
        uncovered = set(all_cc_edges)
        selected_v = []
        
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

        if uncovered:
            return best_cliques, best_count

        # Initialize S_v[v] with cross-edge requirements
        S_v = {v: set() for v in range(k, n)}
        for (u, v) in cross_edges_norm:
            if v in selected_v:
                S_v[v].add(u)

        # Assign remaining C-C edges
        cc_covered = set()
        for v in selected_v:
            nc_v = sorted(S_v[v])
            for i in range(len(nc_v)):
                for j in range(i + 1, len(nc_v)):
                    cc_covered.add((nc_v[i], nc_v[j]))

        remaining_cc = set(all_cc_edges) - cc_covered
        for (u, w) in sorted(remaining_cc):
            if _time_up():
                break
            assigned = False
            for v in selected_v:
                if u in N_C[v] and w in N_C[v]:
                    S_v[v].add(u)
                    S_v[v].add(w)
                    assigned = True
                    break
            if not assigned:
                return best_cliques, best_count

        # Build initial cliques
        cliques = []
        for v in selected_v:
            if S_v[v]:
                clique = sorted([v] + list(S_v[v]))
                cliques.append(clique)

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

        # LOCAL SEARCH: Try to reduce number of I-vertices in selected_v
        # by merging their C-vertex assignments into other I-vertices
        import random as _rnd
        _rng = _rnd.Random(seed + 999)
        
        for iteration in range(100):
            if _time_up():
                break
            
            if len(selected_v) < 2:
                break
            
            # Try to remove one I-vertex by distributing its C-vertices to others
            v_idx = _rng.randrange(len(selected_v))
            v_remove = selected_v[v_idx]
            
            # Find C-vertices that can be moved
            movable = list(S_v[v_remove])
            if not movable:
                continue
            
            # Check if all C-vertices in S_v[v_remove] can be covered by other selected_v
            can_remove = True
            for u in movable:
                found = False
                for v2 in selected_v:
                    if v2 != v_remove and u in N_C[v2]:
                        # Check if all edges (u, w) for w in S_v[v_remove] are covered
                        all_covered = True
                        for w in S_v[v_remove]:
                            if w == u:
                                continue
                            # Edge (u,w) must be covered by some v3
                            covered = False
                            for v3 in selected_v:
                                if v3 != v_remove and u in S_v[v3] and w in S_v[v3]:
                                    covered = True
                                    break
                            if not covered:
                                all_covered = False
                                break
                        if all_covered:
                            found = True
                            break
                if not found:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove v_remove
                selected_v.remove(v_remove)
                # Rebuild S_v from scratch
                S_v_new = {v: set() for v in range(k, n)}
                for (u, v) in cross_edges_norm:
                    if v in selected_v:
                        S_v_new[v].add(u)
                
                cc_covered_new = set()
                for v in selected_v:
                    nc_v = sorted(S_v_new[v])
                    for i in range(len(nc_v)):
                        for j in range(i + 1, len(nc_v)):
                            cc_covered_new.add((nc_v[i], nc_v[j]))
                
                remaining_new = set(all_cc_edges) - cc_covered_new
                for (u, w) in sorted(remaining_new):
                    assigned = False
                    for v in selected_v:
                        if u in N_C[v] and w in N_C[v]:
                            S_v_new[v].add(u)
                            S_v_new[v].add(w)
                            assigned = True
                            break
                    if not assigned:
                        # Roll back
                        selected_v.insert(v_idx, v_remove)
                        S_v = {v: set() for v in range(k, n)}
                        for (u2, v2) in cross_edges_norm:
                            if v2 in selected_v:
                                S_v[v2].add(u2)
                        cc_covered_rb = set()
                        for v2 in selected_v:
                            nc_v2 = sorted(S_v[v2])
                            for i in range(len(nc_v2)):
                                for j in range(i + 1, len(nc_v2)):
                                    cc_covered_rb.add((nc_v2[i], nc_v2[j]))
                        remaining_rb = set(all_cc_edges) - cc_covered_rb
                        for (u2, w2) in sorted(remaining_rb):
                            for v2 in selected_v:
                                if u2 in N_C[v2] and w2 in N_C[v2]:
                                    S_v[v2].add(u2)
                                    S_v[v2].add(w2)
                                    break
                        break
                
                if assigned:
                    S_v = S_v_new
                    # Rebuild cliques
                    cliques = []
                    for v in selected_v:
                        if S_v[v]:
                            clique = sorted([v] + list(S_v[v]))
                            cliques.append(clique)
                    
                    # Verify
                    valid = True
                    for cl in cliques:
                        for i in range(len(cl)):
                            for j in range(i + 1, len(cl)):
                                a, b = cl[i], cl[j]
                                if a > b:
                                    a, b = b, a
                                if (a, b) not in edge_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
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
                                valid = False
                                break
                        
                        if valid and len(cliques) < best_count:
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
