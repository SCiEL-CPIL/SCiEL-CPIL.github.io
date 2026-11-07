# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    from collections import defaultdict

    _T0 = time.time()
    _DEADLINE = 150.0 - 15.0  # safety margin

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

    def _max_clique_tomita(adj, candidates):
        """Find a maximum clique in the subgraph induced by candidates using Tomita's algorithm."""
        if not candidates:
            return []
        candidates = list(candidates)
        best = []
        best_size = 0

        def expand(R, P):
            nonlocal best, best_size
            if not P:
                if len(R) > best_size:
                    best_size = len(R)
                    best = list(R)
                return
            if len(R) + len(P) <= best_size:
                return
            # Pivoting: pick pivot u in P with max degree in P
            pivot = max(P, key=lambda u: len(adj[u] & set(P)))
            candidates_for_pivot = [v for v in P if v not in adj[pivot]]
            # If pivot is isolated in P, all of P are candidates
            if not candidates_for_pivot:
                candidates_for_pivot = list(P)

            for v in candidates_for_pivot:
                R.append(v)
                new_P = [w for w in P if w in adj[v]]
                expand(R, new_P)
                R.pop()
                P = [w for w in P if w != v]
                if len(R) + len(P) <= best_size:
                    return

        expand([], candidates)
        return best

    def _solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)

        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        C = set(range(k))
        I = list(range(k, n))

        # Precompute cross neighborhoods
        N_C = {v: adj[v] & C for v in I}
        N_I = {u: adj[u] & set(I) for u in C}

        uncovered = set(edge_set)
        cliques = []

        def add_clique(verts):
            vs = sorted(verts)
            if len(vs) < 2:
                return 0
            if not _is_clique(adj, vs):
                return 0
            cnt = 0
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    e = (min(vs[i], vs[j]), max(vs[i], vs[j]))
                    if e in uncovered:
                        uncovered.discard(e)
                        cnt += 1
            if cnt > 0:
                cliques.append(vs)
            return cnt

        # Phase 1: Add the full clique C (covers all internal C-C edges)
        if k >= 2:
            add_clique(list(C))

        # Phase 2: Greedy selection with Lagrangian edge-weight guidance
        edge_weights = {}
        for e in uncovered:
            edge_weights[e] = 1.0

        while uncovered and time.time() < _DEADLINE:
            best_clique = None
            best_score = -1

            # Build adjacency restricted to uncovered edges
            # A vertex is "active" if it has at least one uncovered edge
            active = set()
            for e in uncovered:
                active.add(e[0])
                active.add(e[1])

            # Candidate generation:
            # 1. For each active vertex v, find max clique in its uncovered-neighborhood
            # 2. Also try the full C clique if any C-C edges remain

            # Candidate: full C clique
            c_edges_remaining = [(u, v) for u in C for v in C if u < v and (u, v) in uncovered]
            if c_edges_remaining:
                c_list = sorted(C)
                score = sum(edge_weights.get((min(a, b), max(a, b)), 0.0)
                            for a in c_list for b in c_list
                            if a < b and (min(a, b), max(a, b)) in uncovered)
                if score > best_score:
                    best_score = score
                    best_clique = c_list

            # For each vertex in I that has uncovered edges, find max clique in its
            # neighborhood restricted to uncovered edges
            for v in I:
                if v not in active:
                    continue
                # Neighbors of v that have uncovered edge to v
                nbrs_uncovered = [u for u in N_C[v] if (min(u, v), max(u, v)) in uncovered]
                if not nbrs_uncovered:
                    continue
                # The clique must include v and some subset of nbrs_uncovered
                # that forms a clique among themselves (they're all in C, so they
                # form a clique iff they're all in C, which they are)
                # Since all nbrs are in C, any subset is a clique.
                # So the best clique containing v is {v} ∪ nbrs_uncovered
                clique = [v] + sorted(nbrs_uncovered)
                score = sum(edge_weights.get((min(u, v), max(u, v)), 0.0) for u in nbrs_uncovered)
                if score > best_score:
                    best_score = score
                    best_clique = clique

            # For vertices in C, find max clique in their uncovered neighborhood
            # A clique in C's uncovered neighborhood: subset of C that are mutually
            # connected via uncovered edges, plus possibly some I vertices
            # But I vertices only connect to C, not to each other.
            # So a clique containing a C-vertex u can include:
            #   - other C-vertices w where (u,w) is uncovered
            #   - I-vertices v where (u,v) is uncovered AND v is adjacent to all other C-vertices in the clique
            # This is complex. Let's use Tomita on the subgraph induced by active vertices
            # but only for vertices in C that still have uncovered edges.

            # Use Tomita's algorithm on the subgraph of active vertices
            # But this could be expensive. Let's limit to a reasonable subset.
            # Strategy: for each C-vertex with uncovered edges, try to build a clique
            # containing it using Tomita on its uncovered neighborhood.

            c_active = [u for u in C if u in active]
            for u in c_active:
                # Uncovered neighbors of u
                nbrs = [w for w in adj[u] if w in active and (min(u, w), max(u, w)) in uncovered]
                if not nbrs:
                    continue
                # Find max clique in {u} ∪ nbrs using Tomita
                # The subgraph induced by {u} ∪ nbrs where edges are from uncovered
                # Actually, we need a clique in the original graph that covers uncovered edges
                # Let's just use Tomita on the full adjacency restricted to active set
                # But that's too expensive. Instead, use a simpler approach:
                # Try {u} ∪ (all uncovered neighbors of u in C) ∪ (I-vertices adjacent to u and all those C-vertices)
                c_nbrs = [w for w in nbrs if w in C]
                i_nbrs = [w for w in nbrs if w in I]
                # Filter I-vertices: must be adjacent to all C-vertices in c_nbrs
                filtered_i = []
                for v in i_nbrs:
                    ok = True
                    for w in c_nbrs:
                        if w not in adj[v]:
                            ok = False
                            break
                    if ok:
                        filtered_i.append(v)
                clique = [u] + sorted(c_nbrs) + sorted(filtered_i)
                if len(clique) >= 2:
                    # Compute score
                    score = 0.0
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            a, b = clique[i], clique[j]
                            e = (min(a, b), max(a, b))
                            if e in uncovered:
                                score += edge_weights.get(e, 0.0)
                    if score > best_score:
                        best_score = score
                        best_clique = clique

            # Also try Tomita on the full active set for a global max clique
            # This is expensive but can find large cliques we might miss
            if len(active) <= 50 and time.time() < _DEADLINE - 5:
                # Build subgraph adjacency for active vertices
                mc = _max_clique_tomita(adj, list(active))
                if len(mc) >= 2:
                    # Check it's actually a clique in original graph
                    if _is_clique(adj, mc):
                        score = 0.0
                        for i in range(len(mc)):
                            for j in range(i + 1, len(mc)):
                                a, b = mc[i], mc[j]
                                e = (min(a, b), max(a, b))
                                if e in uncovered:
                                    score += edge_weights.get(e, 0.0)
                        if score > best_score:
                            best_score = score
                            best_clique = sorted(mc)

            if best_clique is None or best_score <= 0:
                # Fallback: pick any uncovered edge
                e = next(iter(uncovered))
                best_clique = sorted([e[0], e[1]])

            cnt = add_clique(best_clique)
            if cnt == 0:
                break

            # Update edge weights: boost edges incident to vertices with few remaining
            # uncovered edges (they're "hard" to cover, so prioritize them)
            rem_deg = defaultdict(int)
            for e in uncovered:
                rem_deg[e[0]] += 1
                rem_deg[e[1]] += 1

            for e in list(edge_weights.keys()):
                if e not in uncovered:
                    del edge_weights[e]
                    continue
                u, v = e
                du = rem_deg.get(u, 0)
                dv = rem_deg.get(v, 0)
                boost = 1.0 + 2.0 / (1.0 + min(du, dv))
                edge_weights[e] = edge_weights[e] * boost

        # Cover any remaining edges with 2-cliques
        for e in sorted(uncovered):
            cliques.append(sorted([e[0], e[1]]))
            uncovered.discard(e)

        # Phase 3: Local refinement - try to merge cliques or replace pairs with better solutions
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False

            # Try merging two cliques if their union is a clique and covers the same edges
            for i in range(len(cliques)):
                if improved:
                    break
                for j in range(i + 1, len(cliques)):
                    uni = set(cliques[i]) | set(cliques[j])
                    if len(uni) < 2:
                        continue
                    if not _is_clique(adj, uni):
                        continue

                    def edge_set(cl):
                        s = set()
                        for a in range(len(cl)):
                            for b in range(a + 1, len(cl)):
                                s.add((min(cl[a], cl[b]), max(cl[a], cl[b])))
                        return s

                    ei = edge_set(cliques[i])
                    ej = edge_set(cliques[j])
                    eu = edge_set(sorted(uni))
                    if ei | ej == eu:
                        cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                        cliques.append(sorted(uni))
                        improved = True
                        break

            # Try replacing two cliques with a single larger clique
            if not improved:
                for i in range(len(cliques)):
                    if improved:
                        break
                    for j in range(i + 1, len(cliques)):
                        all_edges_ij = set()
                        for cl in (cliques[i], cliques[j]):
                            for a in range(len(cl)):
                                for b in range(a + 1, len(cl)):
                                    all_edges_ij.add((min(cl[a], cl[b]), max(cl[a], cl[b])))
                        verts_involved = set(cliques[i]) | set(cliques[j])
                        candidate = sorted(verts_involved)
                        if _is_clique(adj, candidate):
                            cand_edges = set()
                            for a in range(len(candidate)):
                                for b in range(a + 1, len(candidate)):
                                    cand_edges.add((min(candidate[a], candidate[b]), max(candidate[a], candidate[b])))
                            if all_edges_ij <= cand_edges:
                                cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                                cliques.append(candidate)
                                improved = True
                                break

        # Final validation
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.add((u, v))

        for cl in cliques:
            assert len(cl) >= 2, f"Clique too small: {cl}"
            assert len(set(cl)) == len(cl), f"Duplicate vertices in clique: {cl}"
            assert _is_clique(adj, cl), f"Not a clique: {cl}"

        seen = set()
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (min(cl[a], cl[b]), max(cl[a], cl[b]))
                    assert e not in seen, f"Edge {e} in two cliques"
                    seen.add(e)
        assert seen == all_edges, f"Missing edges: {all_edges - seen}"

        return [list(map(int, cl)) for cl in cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            out.append([list(e) for e in instance(*spec)])
            continue
        out.append(_solve_one(*spec))

    return out
# EVOLVE-BLOCK-END
