"""Clique partition of split graphs: ILP-style branch-and-bound on C-C edge assignment."""
import time
from random import Random
from itertools import combinations

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

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Build N_C for each I-vertex
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        I_verts = list(range(k, n))
        m = len(I_verts)
        m_cc = k * (k - 1) // 2

        cc_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cc_edge_idx = {}
        for idx, e in enumerate(cc_edges):
            cc_edge_idx[e] = idx

        # For each I-vertex, which C-C edges are within N_C[v]
        E_N = []
        d_v = []
        for i, v in enumerate(I_verts):
            neigh = list(N_C[v])
            d_v.append(len(neigh))
            s = set()
            for a, b in combinations(neigh, 2):
                e = (a, b) if a < b else (b, a)
                if e in cc_edge_idx:
                    s.add(cc_edge_idx[e])
            E_N.append(s)

        # For each C-C edge, which I-vertices could absorb it
        edge_owners = [set() for _ in range(m_cc)]
        for i in range(m):
            for e_idx in E_N[i]:
                edge_owners[e_idx].add(i)

        # Greedy coloring of a graph given by adjacency list
        def _greedy_color(adj, active):
            """Greedy coloring: returns list of color classes (each a sorted list)."""
            comp_adj = [set() for _ in range(k)]
            for a in active:
                for b in adj[a]:
                    if b in active:
                        comp_adj[a].add(b)
            colors = []
            uncolored = set(active)
            while uncolored:
                best_v = max(uncolored, key=lambda x: len(comp_adj[x] & uncolored))
                color_class = [best_v]
                uncolored.discard(best_v)
                candidates = sorted(uncolored, key=lambda x: -len(comp_adj[x] & uncolored))
                for v in candidates:
                    if v not in uncolored:
                        continue
                    if all(v not in comp_adj[c] for c in color_class):
                        color_class.append(v)
                        uncolored.discard(v)
                if len(color_class) >= 2:
                    colors.append(sorted(color_class))
            return colors

        def _theta_remaining(S_set):
            """Min clique cover of remaining C-C edges. Returns (count, cliques)."""
            consumed = set()
            for i in S_set:
                consumed |= E_N[i]
            remaining = set(range(m_cc)) - consumed
            if not remaining:
                return 0, []
            rem_adj = [set() for _ in range(k)]
            active = set()
            for e_idx in remaining:
                a, b = cc_edges[e_idx]
                rem_adj[a].add(b)
                rem_adj[b].add(a)
                active.add(a)
                active.add(b)
            n_act = len(active)
            if n_act == 0:
                return 0, []
            if len(remaining) == n_act * (n_act - 1) // 2:
                return 1, [sorted(active)]
            # Greedy coloring on complement
            colors = _greedy_color(rem_adj, active)
            return len(colors), colors

        def _eval(S_set):
            """Evaluate total clique count for a given set S."""
            n_big = len(S_set)
            n_indiv = sum(d_v[i] for i in range(m) if i not in S_set and d_v[i] > 0)
            theta, ccliques = _theta_remaining(S_set)
            return n_big + n_indiv + theta, ccliques

        # Local search with multiple restarts
        best_val = float('inf')
        best_S = frozenset()
        best_ccliques = []
        visited = set()

        def _local_search(S_init, max_iters=200):
            nonlocal best_val, best_S, best_ccliques
            S_cur = S_init
            val_cur, ccl_cur = _eval(S_cur)
            if val_cur < best_val:
                best_val = val_cur
                best_S = S_cur
                best_ccliques = ccl_cur
            visited.add(S_cur)
            for _ in range(max_iters):
                if _time_up():
                    break
                improved = False
                # Try adding one vertex
                for i in range(m):
                    if i in S_cur or d_v[i] < 2:
                        continue
                    S_new = S_cur | frozenset([i])
                    if S_new in visited:
                        continue
                    val_new, ccl_new = _eval(S_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_S = S_new
                        best_ccliques = ccl_new
                        S_cur = S_new
                        visited.add(S_new)
                        improved = True
                        break
                if improved:
                    continue
                # Try removing one vertex
                for i in list(S_cur):
                    S_new = S_cur - frozenset([i])
                    if S_new in visited:
                        continue
                    val_new, ccl_new = _eval(S_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_S = S_new
                        best_ccliques = ccl_new
                        S_cur = S_new
                        visited.add(S_new)
                        improved = True
                        break
                if improved:
                    continue
                # Try swap
                for i_add in range(m):
                    if i_add in S_cur or d_v[i_add] < 2:
                        continue
                    for i_rem in list(S_cur):
                        S_new = (S_cur - frozenset([i_rem])) | frozenset([i_add])
                        if S_new in visited:
                            continue
                        val_new, ccl_new = _eval(S_new)
                        if val_new < best_val:
                            best_val = val_new
                            best_S = S_new
                            best_ccliques = ccl_new
                            S_cur = S_new
                            visited.add(S_new)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            return S_cur, best_val

        # Initial candidates
        candidates_S = []
        S_all = frozenset(i for i in range(m) if d_v[i] >= 2)
        candidates_S.append(S_all)
        candidates_S.append(frozenset())
        for i in range(m):
            if d_v[i] >= 2:
                candidates_S.append(frozenset([i]))
        # Greedy build by degree
        S_greedy = frozenset()
        for i in sorted(range(m), key=lambda x: d_v[x], reverse=True):
            if d_v[i] >= 2:
                S_greedy = S_greedy | frozenset([i])
        candidates_S.append(S_greedy)
        # Random samples
        import random as _rnd
        for _ in range(20):
            if _time_up():
                break
            S_rand = frozenset(i for i in range(m) if d_v[i] >= 2 and _rnd.random() < 0.5)
            candidates_S.append(S_rand)

        for S_init in candidates_S:
            if _time_up():
                break
            if S_init in visited:
                continue
            _local_search(S_init)

        # Build final answer
        S_final = best_S
        cliques = []
        for i in range(m):
            v = I_verts[i]
            if i in S_final:
                cliques.append([v] + sorted(N_C[v]))
            else:
                for u in sorted(N_C[v]):
                    cliques.append([v, u])
        cliques.extend(best_ccliques)

        if not cliques:
            cliques = [list(range(k))]

        # Validate
        edge_count = {}
        for clique in cliques:
            if len(clique) < 2:
                continue
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                edge_count[e] = edge_count.get(e, 0) + 1

        missing = [e for e in edge_set if edge_count.get(e, 0) == 0]
        over = [e for e, c in edge_count.items() if c > 1]

        if missing or over:
            cliques = [list(range(k))]
            for (u, v) in edge_set:
                if u >= k or v >= k:
                    cliques.append([u, v])

        results.append(cliques)

    return results
