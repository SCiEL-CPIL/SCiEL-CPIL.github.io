import time
from collections import defaultdict
import random as _random
import math

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_bitmask(adj_masks, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting
        and bitmasks for speed.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if R.bit_count() >= 2:
                    # Convert bitmask to frozenset
                    vs = []
                    for v in range(n):
                        if R & (1 << v):
                            vs.append(v)
                    cliques.append(frozenset(vs))
                return
            
            # Choose pivot u in P ∪ X to maximize |P ∩ adj[u]|
            PX = P | X
            best_u = -1
            best_count = -1
            # Find pivot
            temp = PX
            while temp:
                u = (temp & (-temp)).bit_length() - 1
                temp ^= (1 << u)
                cnt = (P & adj_masks[u]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            
            if best_u == -1:
                candidates = P
            else:
                candidates = P & ~adj_masks[best_u]
            
            while candidates:
                if time.time() - start_time > DEADLINE:
                    return
                v_bit = candidates & (-candidates)
                v = v_bit.bit_length() - 1
                candidates ^= v_bit
                
                R_new = R | v_bit
                P_new = P & adj_masks[v]
                X_new = X & adj_masks[v]
                bk(R_new, P_new, X_new)
                
                P &= ~v_bit
                X |= v_bit
        
        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        # Build adjacency as bitmask array
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch_bitmask(adj_masks, n)
        
        if not cliques:
            return []
        
        m = len(cliques)
        
        # Build vertex_cliques: map vertex index to set of clique indices
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)
                
        # Identify active vertices (those in at least one clique)
        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        
        if not active_vertices:
            return []
        
        # Dominance Pruning: If C_u is subset of C_v, u is dominated by v
        active_vertices_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))
        
        keep_vertices = set()
        for v in active_vertices_sorted:
            dominated = False
            for u in keep_vertices:
                if vertex_cliques[v].issubset(vertex_cliques[u]):
                    dominated = True
                    break
            if not dominated:
                keep_vertices.add(v)
                
        candidates = list(keep_vertices)
        
        if not candidates:
            return []

        # Convert vertex_cliques to bitmask for faster intersection
        vertex_clique_masks = [0] * n
        for v in range(n):
            mask = 0
            for ci in vertex_cliques[v]:
                mask |= (1 << ci)
            vertex_clique_masks[v] = mask
        
        # Greedy Set Cover for initial upper bound
        def greedy_cover():
            selected = set()
            unc_mask = (1 << m) - 1
            while unc_mask:
                best_v = -1
                best_cov = 0
                for v in candidates:
                    if v in selected:
                        continue
                    cov = (unc_mask & vertex_clique_masks[v]).bit_count()
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                unc_mask &= ~vertex_clique_masks[best_v]
            return selected

        init_sol = greedy_cover()
        best_size = len(init_sol)
        best_solution = set(init_sol)
        
        # Branch and Bound with bitmask
        def branch_and_bound(selected_mask, uncovered_mask):
            nonlocal best_size, best_solution
            
            if not uncovered_mask:
                sz = selected_mask.bit_count()
                if sz < best_size:
                    best_size = sz
                    # Convert selected_mask to set
                    sol = set()
                    temp = selected_mask
                    while temp:
                        bit = temp & (-temp)
                        v = bit.bit_length() - 1
                        sol.add(v)
                        temp ^= bit
                    best_solution = sol
                return
            
            sz = selected_mask.bit_count()
            if sz >= best_size:
                return
                
            # Lower bound: max coverage of any single candidate vertex
            max_cov = 0
            for v in candidates:
                if selected_mask & (1 << v):
                    continue
                cov = (uncovered_mask & vertex_clique_masks[v]).bit_count()
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return

            lb = sz + math.ceil(uncovered_mask.bit_count() / max_cov)
            if lb >= best_size:
                return
                
            if time.time() - start_time > DEADLINE:
                return

            # Choose a clique to branch on: the one with fewest available vertices
            target_clique_idx = -1
            min_avail = float('inf')
            
            # Iterate through uncovered cliques
            temp_unc = uncovered_mask
            while temp_unc:
                bit = temp_unc & (-temp_unc)
                ci = bit.bit_length() - 1
                temp_unc ^= bit
                
                # Count available vertices in this clique
                avail = 0
                for v in cliques[ci]:
                    if (1 << v) & selected_mask:
                        continue
                    if v in candidates:
                        avail += 1
                if avail < min_avail:
                    min_avail = avail
                    target_clique_idx = ci
                    if min_avail <= 1:
                        break
                    
            if target_clique_idx == -1:
                return
            
            target_clique = cliques[target_clique_idx]
            
            # Branch on vertices in target_clique
            verts_in_target = [v for v in target_clique if v in candidates and not (selected_mask & (1 << v))]
            
            if not verts_in_target:
                return
                
            # Order by coverage descending
            verts_in_target.sort(key=lambda v: -(uncovered_mask & vertex_clique_masks[v]).bit_count())
            
            for v in verts_in_target:
                new_selected = selected_mask | (1 << v)
                new_uncovered = uncovered_mask & ~vertex_clique_masks[v]
                branch_and_bound(new_selected, new_uncovered)
                
        initial_uncovered = (1 << m) - 1
        branch_and_bound(0, initial_uncovered)
        
        # Local search polish: try to remove vertices or swap for better
        if best_solution:
            # Try removing each vertex and see if solution is still valid
            improved = True
            while improved:
                improved = False
                for v in list(best_solution):
                    trial = best_solution - {v}
                    # Check if trial covers all cliques
                    valid = True
                    for c in cliques:
                        if not (c & trial):
                            valid = False
                            break
                    if valid:
                        best_solution = trial
                        best_size -= 1
                        improved = True
                        break
            
            # Try 1-for-1 swaps: remove v, add w, check validity
            for v in list(best_solution):
                for w in candidates:
                    if w in best_solution:
                        continue
                    trial = (best_solution - {v}) | {w}
                    valid = True
                    for c in cliques:
                        if not (c & trial):
                            valid = False
                            break
                    if valid and len(trial) < best_size:
                        best_solution = trial
                        best_size = len(trial)
                        break

        if best_solution is None:
            return []
            
        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)
        
    return results
