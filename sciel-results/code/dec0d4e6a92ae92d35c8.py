"""Clique partition of split graphs: matching + coloring + local search."""
import time
import itertools

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

        # Build cross-neighborhoods: for each i in I, which C-vertices it's connected to
        cross = {i: set() for i in I}
        for (a, b) in edges:
            if a < k and b >= k:
                cross[b].add(a)
            elif b < k and a >= k:
                cross[a].add(b)

        # All possible C-C edges
        all_C_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        covered_C_edges = set()

        cliques = []

        def try_matching(i):
            """Use matching-based pair packing for I-vertex i."""
            neighbors = sorted(cross[i])
            if not neighbors:
                return [], set()
            
            local_covered = set()
            local_cliques = []
            matched = set()
            
            # Compute degrees in remaining C-edges
            def c_deg(c):
                return sum(1 for v in range(k) if v != c and (min(c,v), max(c,v)) in all_C_edges 
                           and (min(c,v), max(c,v)) not in covered_C_edges 
                           and (min(c,v), max(c,v)) not in local_covered)
            
            degs = {c: c_deg(c) for c in neighbors}
            sorted_n = sorted(neighbors, key=lambda c: degs[c])
            
            for idx_a in range(len(sorted_n)):
                if sorted_n[idx_a] in matched:
                    continue
                c_a = sorted_n[idx_a]
                best_partner = None
                best_cost = float('inf')
                for idx_b in range(idx_a + 1, len(sorted_n)):
                    c_b = sorted_n[idx_b]
                    if c_b in matched:
                        continue
                    e = (min(c_a, c_b), max(c_a, c_b))
                    if e not in covered_C_edges and e not in local_covered:
                        cost = degs[c_a] + degs[c_b]
                        if cost < best_cost:
                            best_cost = cost
                            best_partner = c_b
                if best_partner is not None:
                    matched.add(c_a)
                    matched.add(best_partner)
                    local_covered.add((min(c_a, best_partner), max(c_a, best_partner)))
                    local_cliques.append([i, c_a, best_partner])
            
            for c in neighbors:
                if c not in matched:
                    local_cliques.append([i, c])
            
            return local_cliques, local_covered

        def try_coloring(i):
            """Use conflict-graph coloring for I-vertex i."""
            neighbors = sorted(cross[i])
            if not neighbors:
                return [], set()
            
            local_covered = set()
            local_cliques = []
            
            # Build conflict graph: edge (u,w) in neighbors if (u,w) is a C-edge not yet covered
            # We want to partition neighbors into independent sets (no conflicts)
            # Each independent set S gives clique {i} U S
            # Conflict exists if (u,w) is in all_C_edges and not covered and not local_covered
            
            def conflict(u, w):
                e = (min(u, w), max(u, w))
                return e in all_C_edges and e not in covered_C_edges and e not in local_covered
            
            # Greedy coloring: assign each vertex to smallest color where no conflict
            colors = {}
            color_sets = []
            
            # Order by degree in conflict graph (descending)
            def conf_deg(v):
                return sum(1 for w in neighbors if w != v and conflict(v, w))
            
            ordered = sorted(neighbors, key=lambda v: conf_deg(v), reverse=True)
            
            for v in ordered:
                used = set()
                for w in neighbors:
                    if w == v or w not in colors:
                        continue
                    if conflict(v, w):
                        used.add(colors[w])
                c = 0
                while c in used:
                    c += 1
                colors[v] = c
                if c >= len(color_sets):
                    color_sets.append(set())
                color_sets[c].add(v)
            
            for S in color_sets:
                if not S:
                    continue
                S_list = sorted(S)
                local_cliques.append([i] + S_list)
                # Mark all C-edges within S as covered
                for idx, u in enumerate(S_list):
                    for w in S_list[idx+1:]:
                        local_covered.add((min(u, w), max(u, w)))
            
            return local_cliques, local_covered

        # Process I-vertices in ascending degree
        for i in sorted(I, key=lambda x: len(cross[x])):
            if not cross[i]:
                continue
            
            # Try both strategies, pick the one covering more C-edges
            m_cliques, m_covered = try_matching(i)
            c_cliques, c_covered = try_coloring(i)
            
            if len(c_covered) > len(m_covered):
                chosen_cliques = c_cliques
                chosen_covered = c_covered
            else:
                chosen_cliques = m_cliques
                chosen_covered = m_covered
            
            cliques.extend(chosen_cliques)
            covered_C_edges.update(chosen_covered)

        # Cover remaining C-edges with pure-C cliques (greedy maximal clique)
        remaining = all_C_edges - covered_C_edges
        while remaining:
            if time.time() > deadline:
                break
            # Find vertex with highest degree in remaining
            degs = {}
            for (u, w) in remaining:
                degs[u] = degs.get(u, 0) + 1
                degs[w] = degs.get(w, 0) + 1
            if not degs:
                break
            start_v = max(degs, key=lambda x: degs[x])
            
            # Build candidate set: neighbors of start_v in remaining
            candidates = set()
            for (u, w) in remaining:
                if u == start_v:
                    candidates.add(w)
                elif w == start_v:
                    candidates.add(u)
            
            clique = {start_v}
            while candidates:
                # Pick candidate with max connections to current clique
                best_cand = None
                best_score = -1
                for c in candidates:
                    score = sum(1 for u in clique if (min(c, u), max(c, u)) in remaining)
                    if score > best_score:
                        best_score = score
                        best_cand = c
                if best_cand is None or best_score < len(clique):
                    break
                clique.add(best_cand)
                candidates.discard(best_cand)
                # Filter candidates: must connect to all clique members
                new_cands = set()
                for c in candidates:
                    if all((min(c, u), max(c, u)) in remaining for u in clique):
                        new_cands.add(c)
                candidates = new_cands
            
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                # Remove covered edges
                cl = sorted(clique)
                for idx, u in enumerate(cl):
                    for w in cl[idx+1:]:
                        e = (min(u, w), max(u, w))
                        remaining.discard(e)

        # Local search: try to reduce clique count
        # For each pair of cliques, check if they can be merged (union is a clique)
        # For each clique, check if it can be split into two smaller cliques covering same edges
        # Simpler: try replacing 2 cliques with 1 if union is a clique
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    # Check if union of cliques[i] and cliques[j] is a clique
                    union = set(cliques[i]) | set(cliques[j])
                    is_clique = True
                    ul = sorted(union)
                    for idx, u in enumerate(ul):
                        for w in ul[idx+1:]:
                            if (u, w) not in edges_set and (w, u) not in edges_set:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        # Replace two cliques with one
                        cliques[i] = ul
                        cliques.pop(j)
                        improved = True
                        break
                if improved:
                    break

        return [list(c) for c in cliques]

    # Precompute edge set for fast lookup (used in local search)
    # We need to rebuild per instance, so do it inside solve
    # Actually, let's restructure: pass edges_set to solve

    def solve2(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edges_set = set((min(a,b), max(a,b)) for a, b in edges)
        I = list(range(k, n))
        C = list(range(k))

        cross = {i: set() for i in I}
        for (a, b) in edges:
            if a < k and b >= k:
                cross[b].add(a)
            elif b < k and a >= k:
                cross[a].add(b)

        all_C_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        covered_C_edges = set()
        cliques = []

        def try_matching(i):
            neighbors = sorted(cross[i])
            if not neighbors:
                return [], set()
            local_covered = set()
            local_cliques = []
            matched = set()

            def c_deg(c):
                return sum(1 for v in range(k) if v != c and (min(c,v), max(c,v)) in all_C_edges
                           and (min(c,v), max(c,v)) not in covered_C_edges
                           and (min(c,v), max(c,v)) not in local_covered)

            degs = {c: c_deg(c) for c in neighbors}
            sorted_n = sorted(neighbors, key=lambda c: degs[c])

            for idx_a in range(len(sorted_n)):
                if sorted_n[idx_a] in matched:
                    continue
                c_a = sorted_n[idx_a]
                best_partner = None
                best_cost = float('inf')
                for idx_b in range(idx_a + 1, len(sorted_n)):
                    c_b = sorted_n[idx_b]
                    if c_b in matched:
                        continue
                    e = (min(c_a, c_b), max(c_a, c_b))
                    if e not in covered_C_edges and e not in local_covered:
                        cost = degs[c_a] + degs[c_b]
                        if cost < best_cost:
                            best_cost = cost
                            best_partner = c_b
                if best_partner is not None:
                    matched.add(c_a)
                    matched.add(best_partner)
                    local_covered.add((min(c_a, best_partner), max(c_a, best_partner)))
                    local_cliques.append([i, c_a, best_partner])

            for c in neighbors:
                if c not in matched:
                    local_cliques.append([i, c])

            return local_cliques, local_covered

        def try_coloring(i):
            neighbors = sorted(cross[i])
            if not neighbors:
                return [], set()
            local_covered = set()
            local_cliques = []

            def conflict(u, w):
                e = (min(u, w), max(u, w))
                return e in all_C_edges and e not in covered_C_edges and e not in local_covered

            def conf_deg(v):
                return sum(1 for w in neighbors if w != v and conflict(v, w))

            ordered = sorted(neighbors, key=lambda v: conf_deg(v), reverse=True)
            colors = {}
            color_sets = []

            for v in ordered:
                used = set()
                for w in neighbors:
                    if w == v or w not in colors:
                        continue
                    if conflict(v, w):
                        used.add(colors[w])
                c = 0
                while c in used:
                    c += 1
                colors[v] = c
                if c >= len(color_sets):
                    color_sets.append(set())
                color_sets[c].add(v)

            for S in color_sets:
                if not S:
                    continue
                S_list = sorted(S)
                local_cliques.append([i] + S_list)
                for idx, u in enumerate(S_list):
                    for w in S_list[idx+1:]:
                        local_covered.add((min(u, w), max(u, w)))

            return local_cliques, local_covered

        for i in sorted(I, key=lambda x: len(cross[x])):
            if not cross[i]:
                continue
            m_cliques, m_covered = try_matching(i)
            c_cliques, c_covered = try_coloring(i)
            if len(c_covered) > len(m_covered):
                chosen_cliques, chosen_covered = c_cliques, c_covered
            else:
                chosen_cliques, chosen_covered = m_cliques, m_covered
            cliques.extend(chosen_cliques)
            covered_C_edges.update(chosen_covered)

        remaining = all_C_edges - covered_C_edges
        while remaining:
            if time.time() > deadline:
                break
            degs = {}
            for (u, w) in remaining:
                degs[u] = degs.get(u, 0) + 1
                degs[w] = degs.get(w, 0) + 1
            if not degs:
                break
            start_v = max(degs, key=lambda x: degs[x])
            candidates = set()
            for (u, w) in remaining:
                if u == start_v:
                    candidates.add(w)
                elif w == start_v:
                    candidates.add(u)
            clique = {start_v}
            while candidates:
                best_cand = None
                best_score = -1
                for c in candidates:
                    score = sum(1 for u in clique if (min(c, u), max(c, u)) in remaining)
                    if score > best_score:
                        best_score = score
                        best_cand = c
                if best_cand is None or best_score < len(clique):
                    break
                clique.add(best_cand)
                candidates.discard(best_cand)
                new_cands = set()
                for c in candidates:
                    if all((min(c, u), max(c, u)) in remaining for u in clique):
                        new_cands.add(c)
                candidates = new_cands
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                cl = sorted(clique)
                for idx, u in enumerate(cl):
                    for w in cl[idx+1:]:
                        e = (min(u, w), max(u, w))
                        remaining.discard(e)

        # Local search: merge cliques if union is a clique
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    union = set(cliques[i]) | set(cliques[j])
                    is_clique = True
                    ul = sorted(union)
                    for idx, u in enumerate(ul):
                        for w in ul[idx+1:]:
                            if (u, w) not in edges_set:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        cliques[i] = ul
                        cliques.pop(j)
                        improved = True
                        break
                if improved:
                    break

        return [list(c) for c in cliques]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve2(*spec))
    return results
# EVOLVE-BLOCK-END
