"""Lagrangian relaxation clique packing: dynamic edge weights guide greedy maximal clique selection."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    from collections import defaultdict

    _T0 = time.time()
    _DEADLINE = 150.0 - 10.0  # safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _is_clique(adj, verts):
        vs = list(verts)
        for i in range(len(vs)):
            a = vs[i]
            for j in range(i + 1, len(vs)):
                if vs[j] not in adj[a]:
                    return False
        return True

    def _solve_one(n, k, p, seed):
        adj = {v: set() for v in range(n)}
        for (u, v) in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)

        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.add((u, v))

        # Lagrangian weights for each edge (dual variables)
        edge_weight = {e: 1.0 for e in all_edges}
        # Multiplier for number-of-cliques penalty
        lambda_c = 1.0

        def clique_weight(cl):
            """Total weight of edges in clique minus lambda penalty."""
            w = 0.0
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    e = (cl[i], cl[j])
                    w += edge_weight.get(e, 1.0)
            return w - lambda_c

        def greedy_max_weight_clique(res_adj, res_edges):
            """Find a maximal clique maximizing total edge weight in residual graph."""
            if not res_edges:
                return []
            # Score each vertex by sum of weights of incident residual edges
            vertex_score = defaultdict(float)
            for (u, v) in res_edges:
                w = edge_weight.get((u, v), 1.0)
                vertex_score[u] += w
                vertex_score[v] += w

            # Greedy: start from highest-score vertex, expand by adding
            # vertices that maintain clique property and maximize marginal gain
            best_clique = []
            best_gain = -1

            # Try starting from top-scoring vertices
            candidates = sorted(vertex_score.keys(), key=lambda v: -vertex_score[v])
            for start in candidates[:min(20, len(candidates))]:
                if not res_adj[start]:
                    continue
                # Build clique greedily
                clique = [start]
                remaining = set(res_adj[start])
                while remaining:
                    # Find vertex with max marginal gain that connects to all in clique
                    best_v = None
                    best_marginal = 0
                    for v in remaining:
                        # Check v connects to all in clique
                        if all(cl in adj[v] for cl in clique):
                            # Marginal: sum of edge weights from v to clique
                            marginal = sum(edge_weight.get((min(cl, v), max(cl, v)), 1.0) for cl in clique)
                            # Also consider future potential
                            marginal += 0.5 * sum(
                                edge_weight.get((min(v, u), max(v, u)), 1.0)
                                for u in remaining if u != v and all(cl in adj[u] for cl in clique)
                            )
                            if marginal > best_marginal:
                                best_marginal = marginal
                                best_v = v
                    if best_v is None:
                        break
                    clique.append(best_v)
                    remaining.discard(best_v)
                    # Update remaining: only keep vertices adjacent to all in clique
                    remaining = {v for v in remaining if all(cl in adj[v] for cl in clique)}

                if len(clique) >= 2:
                    gain = clique_weight(clique)
                    if gain > best_gain:
                        best_gain = gain
                        best_clique = clique

            return best_clique

        # Main Lagrangian loop
        cliques = []
        covered = set()
        res_adj = {v: set(adj[v]) for v in range(n)}
        res_edges = set(all_edges)

        # Initial: cover the large clique C if it's beneficial
        C = list(range(k))
        if _is_clique(adj, C):
            cliques.append(C)
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    e = (C[i], C[j])
                    covered.add(e)
                    res_edges.discard(e)
            # Update res_adj
            for v in C:
                res_adj[v] = set()
            for (u, v) in list(res_edges):
                if u in C or v in C:
                    res_edges.discard((u, v))
                    res_adj[u].discard(v)
                    res_adj[v].discard(u)

        # Lagrangian iterations
        max_iter = 200
        for iteration in range(max_iter):
            if time.time() > _DEADLINE:
                break
            if not res_edges:
                break

            # Find greedy max-weight clique
            cl = greedy_max_weight_clique(res_adj, res_edges)
            if len(cl) < 2:
                # No clique found; break into edges
                for (u, v) in list(res_edges):
                    cliques.append([u, v])
                    covered.add((u, v))
                    res_adj[u].discard(v)
                    res_adj[v].discard(u)
                res_edges.clear()
                break

            # Add this clique
            cliques.append(sorted(cl))
            cl_set = set(cl)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    e = (cl[i], cl[j])
                    if e in res_edges:
                        covered.add(e)
                        res_edges.discard(e)
            # Update residual
            for v in cl:
                res_adj[v].clear()
            for (u, v) in list(res_edges):
                if u in cl_set or v in cl_set:
                    res_edges.discard((u, v))
                    res_adj[u].discard(v)
                    res_adj[v].discard(u)

            # Update Lagrangian multipliers: increase weight of uncovered edges
            # to encourage covering them in subsequent iterations
            for e in res_edges:
                edge_weight[e] = edge_weight.get(e, 1.0) * 1.15  # increase weight
            # Slightly increase lambda to penalize adding more cliques
            lambda_c = lambda_c * 1.02

            # Periodically: try to merge cliques if union is a clique
            if iteration % 10 == 9:
                improved = True
                while improved and time.time() < _DEADLINE:
                    improved = False
                    for i in range(len(cliques)):
                        for j in range(i + 1, len(cliques)):
                            uni = set(cliques[i]) | set(cliques[j])
                            if len(uni) >= 2 and _is_clique(adj, uni):
                                # Check edge coverage
                                ei = set()
                                for a in range(len(cliques[i])):
                                    for b in range(a + 1, len(cliques[i])):
                                        ei.add((cliques[i][a], cliques[i][b]))
                                ej = set()
                                for a in range(len(cliques[j])):
                                    for b in range(a + 1, len(cliques[j])):
                                        ej.add((cliques[j][a], cliques[j][b]))
                                eu = set()
                                for a in range(len(uni)):
                                    for b in range(a + 1, len(uni)):
                                        eu.add((sorted(uni)[a], sorted(uni)[b]))
                                eu = set()
                                uni_sorted = sorted(uni)
                                for a in range(len(uni_sorted)):
                                    for b in range(a + 1, len(uni_sorted)):
                                        eu.add((uni_sorted[a], uni_sorted[b]))
                                if ei | ej == eu:
                                    cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                                    cliques.append(uni_sorted)
                                    improved = True
                                    break
                        if improved:
                            break

        # Safety: cover any remaining edges
        for (u, v) in list(res_edges):
            cliques.append([u, v])
            covered.add((u, v))
        res_edges.clear()

        # Final merge pass
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    uni = set(cliques[i]) | set(cliques[j])
                    if len(uni) >= 2 and _is_clique(adj, uni):
                        uni_sorted = sorted(uni)
                        ei = set()
                        for a in range(len(cliques[i])):
                            for b in range(a + 1, len(cliques[i])):
                                ei.add((cliques[i][a], cliques[i][b]))
                        ej = set()
                        for a in range(len(cliques[j])):
                            for b in range(a + 1, len(cliques[j])):
                                ej.add((cliques[j][a], cliques[j][b]))
                        eu = set()
                        for a in range(len(uni_sorted)):
                            for b in range(a + 1, len(uni_sorted)):
                                eu.add((uni_sorted[a], uni_sorted[b]))
                        if ei | ej == eu:
                            cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                            cliques.append(uni_sorted)
                            improved = True
                            break
                if improved:
                    break

        # Final validation
        for cl in cliques:
            assert len(cl) >= 2
            assert len(set(cl)) == len(cl)
            assert _is_clique(adj, cl)
        seen = set()
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (cl[a], cl[b])
                    assert e not in seen, "edge in two cliques"
                    seen.add(e)
        assert seen == all_edges, "edge not covered"

        return [list(map(int, cl)) for cl in cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            out.append([list(e) for e in instance(*spec)])
            continue
        out.append(_solve_one(*spec))

    return out
# EVOLVE-BLOCK-END
