"""Clique partition of split graphs: set-cover selection + DSATUR coloring of residual."""
import time
from random import Random
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

    def _dsatur_color(adj, vertices):
        """DSATUR coloring of subgraph induced by vertices. Returns list of color classes."""
        n = len(vertices)
        vset = set(vertices)
        # Build local adjacency
        local_adj = {v: set() for v in vertices}
        for v in vertices:
            for u in adj[v]:
                if u in vset:
                    local_adj[v].add(u)
        # DSATUR
        color_of = {}
        uncolored = list(vertices)
        sat = {v: set() for v in vertices}  # colors of neighbors
        while uncolored:
            # Pick vertex with max saturation, tie-break by degree in uncolored
            best = max(uncolored, key=lambda v: (len(sat[v]), len(local_adj[v] & set(uncolored))))
            uncolored.remove(best)
            c = 0
            while c in sat[best]:
                c += 1
            color_of[best] = c
            for u in local_adj[best]:
                if u in set(uncolored):
                    sat[u].add(c)
        # Group by color
        from collections import defaultdict
        groups = defaultdict(list)
        for v, c in color_of.items():
            groups[c].append(v)
        return [sorted(groups[c]) for c in sorted(groups.keys())]

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

        # N_C[v] for each I-vertex
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

        # E_N[i] = set of C-C edge indices covered by I-vertex i
        E_N = []
        d_v = []
        for i, v in enumerate(I_verts):
            neigh = sorted(N_C[v])
            d_v.append(len(neigh))
            s = set()
            for a, b in combinations(neigh, 2):
                e = (a, b) if a < b else (b, a)
                if e in cc_edge_idx:
                    s.add(cc_edge_idx[e])
            E_N.append(s)

        # Residual C-C edge coverage: given selected set S, which C-C edges remain?
        def _residual_edges(S):
            consumed = set()
            for i in S:
                consumed |= E_N[i]
            return set(range(m_cc)) - consumed

        def _color_residual(remaining):
            """Color the graph of remaining C-C edges. Returns list of cliques."""
            if not remaining:
                return []
            active = set()
            adj = [set() for _ in range(k)]
            for e_idx in remaining:
                a, b = cc_edges[e_idx]
                adj[a].add(b)
                adj[b].add(a)
                active.add(a)
                active.add(b)
            if not active:
                return []
            # Check if it's a complete graph
            if len(remaining) == len(active) * (len(active) - 1) // 2:
                return [sorted(active)]
            # DSATUR on complement: color classes of complement = cliques of original
            # Actually: we want to cover edges with cliques. Each clique in original = independent set in complement.
            # So: color the complement graph. Each color class is an independent set in complement = clique in original.
            comp_adj = [set() for _ in range(k)]
            for a in active:
                for b in active:
                    if a != b and b not in adj[a]:
                        comp_adj[a].add(b)
            colors = _dsatur_color(comp_adj, sorted(active))
            # Filter out singletons (they don't cover any edge)
            result = [c for c in colors if len(c) >= 2]
            return result

        def _eval(S):
            """Total clique count for selected set S."""
            n_big = len(S)
            # Cross edges for non-selected I-vertices
            n_indiv = 0
            for i in range(m):
                if i not in S and d_v[i] > 0:
                    n_indiv += d_v[i]
            remaining = _residual_edges(S)
            cliques_rem = _color_residual(remaining)
            theta = len(cliques_rem)
            return n_big + n_indiv + theta, cliques_rem

        # Multi-start greedy set-cover with different strategies
        best_val = float('inf')
        best_S = frozenset()
        best_ccliques = []

        def _try_solution(S):
            nonlocal best_val, best_S, best_ccliques
            val, ccl = _eval(S)
            if val < best_val:
                best_val = val
                best_S = S
                best_ccliques = ccl

        # Strategy 1: Greedy by coverage (most C-C edges covered per vertex)
        def _greedy_cover(order_key):
            uncovered = set(range(m_cc))
            S = frozenset()
            while uncovered and not _time_up():
                best_i = None
                best_key = None
                for i in range(m):
                    if i in S or d_v[i] < 2:
                        continue
                    key = order_key(i, uncovered)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_i = i
                if best_i is None:
                    break
                S = S | frozenset([best_i])
                uncovered -= E_N[best_i]
            return S

        # Different ordering keys
        S1 = _greedy_cover(lambda i, unc: len(E_N[i] & unc))
        _try_solution(S1)

        S2 = _greedy_cover(lambda i, unc: len(E_N[i] & unc) / max(1, d_v[i]))
        _try_solution(S2)

        # Strategy 3: All vertices with d_v >= 2
        S_all = frozenset(i for i in range(m) if d_v[i] >= 2)
        _try_solution(S_all)

        # Strategy 4: Empty set (just color all C-C edges)
        _try_solution(frozenset())

        # Strategy 5: Greedy with random tie-breaking
        rng = Random(42)
        for restart in range(30):
            if _time_up():
                break
            uncovered = set(range(m_cc))
            S = frozenset()
            while uncovered and not _time_up():
                candidates = [i for i in range(m) if i not in S and d_v[i] >= 2 and E_N[i] & uncovered]
                if not candidates:
                    break
                # Weighted random selection
                weights = [len(E_N[i] & unc) for i in candidates]
                total_w = sum(weights)
                if total_w == 0:
                    break
                r = rng.random() * total_w
                cum = 0
                chosen = candidates[-1]
                for idx, i in enumerate(candidates):
                    cum += weights[idx]
                    if r <= cum:
                        chosen = i
                        break
                S = S | frozenset([chosen])
                uncovered -= E_N[chosen]
            _try_solution(S)

        # Local refinement: try adding/removing/swapping one vertex
        def _refine(S, max_rounds=50):
            nonlocal best_val, best_S, best_ccliques
            for _ in range(max_rounds):
                if _time_up():
                    break
                improved = False
                # Try adding
                for i in range(m):
                    if i in S or d_v[i] < 2:
                        continue
                    S_new = S | frozenset([i])
                    val, ccl = _eval(S_new)
                    if val < best_val:
                        best_val = val
                        best_S = S_new
                        best_ccliques = ccl
                        S = S_new
                        improved = True
                        break
                if improved:
                    continue
                # Try removing
                for i in list(S):
                    S_new = S - frozenset([i])
                    val, ccl = _eval(S_new)
                    if val < best_val:
                        best_val = val
                        best_S = S_new
                        best_ccliques = ccl
                        S = S_new
                        improved = True
                        break
                if improved:
                    continue
                # Try swap
                for i_add in range(m):
                    if i_add in S or d_v[i_add] < 2:
                        continue
                    for i_rem in list(S):
                        S_new = (S - frozenset([i_rem])) | frozenset([i_add])
                        val, ccl = _eval(S_new)
                        if val < best_val:
                            best_val = val
                            best_S = S_new
                            best_ccliques = ccl
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            return S

        _refine(best_S)

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
