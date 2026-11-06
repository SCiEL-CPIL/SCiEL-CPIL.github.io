import time
import math
import random as _random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the verifier."""
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj_bitmask(edges, n):
        """Build adjacency as integer bitmasks."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch_bitmask(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting,
        using integer bitmasks for speed.
        Returns a list of lists (each a list of vertex indices).
        """
        cliques = []
        full_mask = (1 << n) - 1

        def popcount(x):
            return bin(x).count('1')

        def bk(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                if popcount(R_mask) >= 2:
                    # Convert bitmask to list of vertex indices
                    verts = [i for i in range(n) if (R_mask >> i) & 1]
                    cliques.append(verts)
                return

            P_union_X = P_mask | X_mask
            if P_union_X == 0:
                return

            # Choose pivot u in P ∪ X to maximize |P ∩ N(u)|
            max_inter = -1
            pivot = -1
            temp = P_union_X
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                inter = P_mask & adj[u]
                cnt = popcount(inter)
                if cnt > max_inter:
                    max_inter = cnt
                    pivot = u
                temp ^= lsb

            candidates = P_mask & ~adj[pivot] if pivot >= 0 else P_mask

            if candidates == 0:
                return

            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1

                new_R = R_mask | lsb
                new_P = P_mask & adj[v]
                new_X = X_mask & adj[v]

                bk(new_R, new_P, new_X)

                P_mask &= ~lsb
                X_mask |= lsb
                temp ^= lsb

        bk(0, full_mask, 0)
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        adj = build_adj_bitmask(edges, n)

        # Enumerate maximal cliques
        cliques = bron_kerbosch_bitmask(adj, n)

        if not cliques:
            return []

        m = len(cliques)

        # Build vertex -> clique coverage bitmask
        vertex_clique_mask = [0] * n
        for ci, c in enumerate(cliques):
            bit = 1 << ci
            for v in c:
                vertex_clique_mask[v] |= bit

        # Identify candidate vertices (those in at least one clique)
        candidates = [v for v in range(n) if vertex_clique_mask[v] != 0]

        if not candidates:
            return []

        # Build the "clique-intersection graph":
        # Nodes are candidate vertices; edge (u,v) exists if u and v co-occur in some maximal clique.
        # A transversal (hitting set) must pick at most one vertex from each maximal clique.
        # So the transversal is an independent set in this intersection graph.
        # We want the MINIMUM transversal = MINIMUM independent set that hits all cliques.
        # But wait: we need to hit ALL cliques, not just pick an independent set.
        #
        # Actually, let me reconsider. The problem is: find minimum set S such that for every
        # maximal clique C (size >= 2), |S ∩ C| >= 1.
        #
        # This is the "clique transversal" or "clique hitting set" problem.
        # It's equivalent to: in the intersection graph (where vertices are original vertices,
        # edges connect vertices in same clique), find a minimum set that is NOT independent
        # in the sense of... hmm, this isn't directly MIS.
        #
        # Let me think differently. The complement approach:
        # - In the intersection graph H (nodes = candidates, edge if co-occur in a clique),
        #   a transversal S must have at least one vertex from each clique.
        # - This is NOT simply an independent set problem.
        #
        # However, we can use the following insight:
        # - The minimum transversal can be found via set cover (which we already do).
        # - But we can improve the branch-and-bound by using the intersection graph structure.
        #
        # Alternative: Use the fact that if two vertices are in the same clique, picking one
        # "covers" that clique. We can model this as a maximum weight independent set in a
        # modified graph, but that's complex.
        #
        # Let me stick with the set cover approach but improve the B&B with better bounds
        # using the intersection graph structure.
        #
        # Build intersection adjacency for candidates
        cand_idx = {v: i for i, v in enumerate(candidates)}
        k = len(candidates)
        
        # intersection_adj[i] = bitmask of candidate indices j such that candidates[i] and candidates[j]
        # are in at least one common clique
        intersection_adj = [0] * k
        for ci, c in enumerate(cliques):
            cand_in_clique = [cand_idx[v] for v in c if v in cand_idx]
            for i in range(len(cand_in_clique)):
                for j in range(i+1, len(cand_in_clique)):
                    intersection_adj[cand_in_clique[i]] |= (1 << cand_in_clique[j])
                    intersection_adj[cand_in_clique[j]] |= (1 << cand_in_clique[i])

        # Dominance Pruning using bitmasks
        active = sorted(candidates, key=lambda v: -bin(vertex_clique_mask[v]).count('1'))

        keep = set()
        for v in active:
            dominated = False
            for u in keep:
                if (vertex_clique_mask[v] & ~vertex_clique_mask[u]) == 0:
                    dominated = True
                    break
            if not dominated:
                keep.add(v)

        candidates = list(keep)
        if not candidates:
            return []

        # Rebuild mapping after pruning
        cand_idx = {v: i for i, v in enumerate(candidates)}
        k = len(candidates)
        vertex_clique_mask_local = [vertex_clique_mask[v] for v in candidates]

        # Greedy Set Cover
        def greedy_cover():
            selected = set()
            unc = (1 << m) - 1
            while unc:
                best_v = -1
                best_cov = 0
                for v in candidates:
                    if v in selected:
                        continue
                    cov = bin(unc & vertex_clique_mask[v]).count('1')
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                unc &= ~vertex_clique_mask[best_v]
            return selected

        init_sol = greedy_cover()
        best_size = len(init_sol)
        best_solution = set(init_sol)

        # Precompute clique candidate lists (as indices into candidates)
        clique_cands = []
        for ci, c in enumerate(cliques):
            cc = [cand_idx[v] for v in c if v in cand_idx]
            clique_cands.append(cc)

        # Improved B&B with intersection graph aware lower bound
        def branch_and_bound(selected_set, uncovered_mask):
            nonlocal best_size, best_solution

            if uncovered_mask == 0:
                if len(selected_set) < best_size:
                    best_size = len(selected_set)
                    best_solution = set(selected_set)
                return

            if len(selected_set) >= best_size:
                return

            if time.time() - start_time > DEADLINE:
                return

            # Lower bound: use packing argument
            # Find a set of cliques that are pairwise "disjoint" in terms of available candidates
            # i.e., no single vertex can cover two of them
            max_cov = 0
            for v in candidates:
                if v in selected_set:
                    continue
                cov = bin(uncovered_mask & vertex_clique_mask[v]).count('1')
                if cov > max_cov:
                    max_cov = cov

            if max_cov == 0:
                return

            # Simple LB
            lb = len(selected_set) + math.ceil(bin(uncovered_mask).count('1') / max_cov)
            if lb >= best_size:
                return

            # Better LB: greedy packing of uncovered cliques
            # Try to find cliques that require distinct vertices
            used_vertices = set()
            pack_count = 0
            temp_unc = uncovered_mask
            while temp_unc:
                lsb = temp_unc & (-temp_unc)
                ci = lsb.bit_length() - 1
                # Find an available vertex for this clique not used yet
                found = False
                for vi in clique_cands[ci]:
                    v = candidates[vi]
                    if v not in selected_set and v not in used_vertices:
                        used_vertices.add(v)
                        found = True
                        break
                if found:
                    pack_count += 1
                temp_unc ^= lsb

            if len(selected_set) + pack_count >= best_size:
                return

            # Branching: choose uncovered clique with fewest available candidates
            target_ci = -1
            min_avail = float('inf')
            temp = uncovered_mask
            while temp:
                lsb = temp & (-temp)
                ci = lsb.bit_length() - 1
                avail = 0
                for vi in clique_cands[ci]:
                    v = candidates[vi]
                    if v not in selected_set:
                        avail += 1
                if avail < min_avail:
                    min_avail = avail
                    target_ci = ci
                    if avail == 1:
                        break
                temp ^= lsb

            if target_ci == -1:
                return

            verts_in_target = [candidates[vi] for vi in clique_cands[target_ci] if candidates[vi] not in selected_set]
            verts_in_target.sort(key=lambda v: -bin(uncovered_mask & vertex_clique_mask[v]).count('1'))

            for v in verts_in_target:
                new_selected = selected_set | {v}
                new_uncovered = uncovered_mask & ~vertex_clique_mask[v]
                branch_and_bound(new_selected, new_uncovered)

        initial_uncovered = (1 << m) - 1
        branch_and_bound(set(), initial_uncovered)

        # Local search polish
        def local_search_polish(sol):
            nonlocal best_size, best_solution
            sol = set(sol)
            improved = True
            while improved:
                improved = False
                for v in list(sol):
                    if len(sol) - 1 >= best_size:
                        continue
                    new_sol = sol - {v}
                    covered = 0
                    for u in new_sol:
                        covered |= vertex_clique_mask[u]
                    if covered == (1 << m) - 1:
                        sol = new_sol
                        improved = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_solution = set(sol)
                        break
            return sol

        if best_solution:
            polished = local_search_polish(best_solution)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = set(polished)

        if best_solution is None:
            return sorted(init_sol)

        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
