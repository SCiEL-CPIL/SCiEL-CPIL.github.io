import time
import math
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        cliques = []
        
        def bk(R, P, X):
            if time.time() - start_time > DEADLINE:
                return
            
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            
            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])
            
            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        m = len(cliques)
        
        # Build vertex_cliques: map vertex index to bitmask of clique indices
        vertex_cliques = [0] * n
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v] |= (1 << ci)
        
        all_cliques_mask = (1 << m) - 1
        
        # Identify active vertices
        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        
        if not active_vertices:
            return []
        
        # Dominance pruning: if v's clique set is subset of u's, v is dominated
        active_sorted = sorted(active_vertices, key=lambda v: -bin(vertex_cliques[v]).count('1'))
        
        keep_vertices = set()
        for v in active_sorted:
            dominated = False
            for u in keep_vertices:
                if (vertex_cliques[v] & ~vertex_cliques[u]) == 0:
                    dominated = True
                    break
            if not dominated:
                keep_vertices.add(v)
        
        candidates = sorted(keep_vertices)
        if not candidates:
            return []
        
        num_cand = len(candidates)
        cand_covers = [vertex_cliques[v] for v in candidates]
        
        # For each clique, which candidates cover it (as bitmask)
        clique_cand_mask = [0] * m
        for i in range(num_cand):
            mask = cand_covers[i]
            for ci in range(m):
                if mask & (1 << ci):
                    clique_cand_mask[ci] |= (1 << i)
        
        # Try ILP first
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            A = np.zeros((m, num_cand))
            for ci in range(m):
                for i in range(num_cand):
                    if A[ci, i] == 0 and (cand_covers[i] & (1 << ci)):
                        A[ci, i] = 1.0
            
            c = np.ones(num_cand)
            constraints = [LinearConstraint(A, lb=np.ones(m), ub=np.inf)]
            bounds = Bounds(lb=0, ub=1)
            
            result = milp(c=c, constraints=constraints, bounds=bounds, 
                         integrality=np.ones(num_cand))
            
            if result.success and result.x is not None:
                sol_mask = 0
                for i in range(num_cand):
                    if result.x[i] > 0.5:
                        sol_mask |= (1 << i)
                # Verify
                covered = 0
                for i in range(num_cand):
                    if sol_mask & (1 << i):
                        covered |= cand_covers[i]
                if covered == all_cliques_mask:
                    best_vertices = [candidates[i] for i in range(num_cand) if sol_mask & (1 << i)]
                    return sorted(best_vertices)
        except Exception:
            pass
        
        # Greedy for initial upper bound
        def greedy_cover(uncovered_mask, rng=None):
            selected_mask = 0
            unc = uncovered_mask
            while unc:
                best_idx = -1
                best_cov = 0
                best_list = []
                for i in range(num_cand):
                    if selected_mask & (1 << i):
                        continue
                    cov = bin(unc & cand_covers[i]).count('1')
                    if cov > best_cov:
                        best_cov = cov
                        best_list = [i]
                    elif cov == best_cov and best_cov > 0:
                        best_list.append(i)
                if not best_list:
                    break
                if rng and len(best_list) > 1:
                    best_idx = rng.choice(best_list)
                else:
                    best_idx = best_list[0]
                selected_mask |= (1 << best_idx)
                unc &= ~cand_covers[best_idx]
            return selected_mask

        init_sol = greedy_cover(all_cliques_mask)
        best_size = bin(init_sol).count('1')
        best_mask = init_sol
        
        if best_size == 0:
            return []
        
        # Random restarts
        rng = _random.Random(42)
        for _ in range(20):
            sol = greedy_cover(all_cliques_mask, rng)
            sz = bin(sol).count('1')
            if sz < best_size:
                best_size = sz
                best_mask = sol
        
        # Branch and bound with bitmasks
        def compute_lb(uncovered_mask, selected_count):
            if not uncovered_mask:
                return 0
            # Greedy packing of disjoint cliques
            lb = 0
            remaining = uncovered_mask
            while remaining:
                best_ci = -1
                best_avail = 0
                for ci in range(m):
                    if remaining & (1 << ci):
                        avail = bin(clique_cand_mask[ci] & ~(best_mask & 0)).count('1')  # simplified
                        if avail > best_avail:
                            best_avail = avail
                            best_ci = ci
                if best_ci == -1:
                    break
                lb += 1
                # Remove all cliques covered by any candidate that covers best_ci
                # This is approximate; just remove best_ci
                remaining &= ~(1 << best_ci)
            
            # Also use max coverage bound
            max_cov = 0
            for i in range(num_cand):
                if best_mask & (1 << i):
                    continue
                cov = bin(uncovered_mask & cand_covers[i]).count('1')
                if cov > max_cov:
                    max_cov = cov
            if max_cov == 0:
                return float('inf')
            pack_lb = lb
            cov_lb = math.ceil(bin(uncovered_mask).count('1') / max_cov)
            return max(pack_lb, cov_lb)

        def branch_and_bound(selected_mask, uncovered_mask):
            nonlocal best_size, best_mask
            
            if not uncovered_mask:
                sz = bin(selected_mask).count('1')
                if sz < best_size:
                    best_size = sz
                    best_mask = selected_mask
                return
            
            if bin(selected_mask).count('1') >= best_size:
                return
            
            if time.time() - start_time > DEADLINE:
                return
            
            lb = bin(selected_mask).count('1') + compute_lb(uncovered_mask, selected_mask)
            if lb >= best_size:
                return
            
            # Choose target clique with fewest available candidates
            target_ci = -1
            min_avail = float('inf')
            for ci in range(m):
                if uncovered_mask & (1 << ci):
                    avail = bin(clique_cand_mask[ci] & ~selected_mask).count('1')
                    if avail == 0:
                        return  # infeasible
                    if avail < min_avail:
                        min_avail = avail
                        target_ci = ci
                        if min_avail == 1:
                            break
            
            if target_ci == -1:
                return
            
            # Order candidates by coverage
            avail_list = []
            for i in range(num_cand):
                if (clique_cand_mask[target_ci] >> i) & 1 and not (selected_mask >> i) & 1:
                    cov = bin(uncovered_mask & cand_covers[i]).count('1')
                    avail_list.append((cov, i))
            avail_list.sort(reverse=True)
            
            for cov, i in avail_list:
                new_sel = selected_mask | (1 << i)
                new_unc = uncovered_mask & ~cand_covers[i]
                branch_and_bound(new_sel, new_unc)
                if time.time() - start_time > DEADLINE - 5:
                    return
        
        branch_and_bound(0, all_cliques_mask)
        
        # Local search: remove redundant vertices
        def local_search(sol_mask):
            improved = True
            while improved:
                improved = False
                # Single removal
                for i in range(num_cand):
                    if sol_mask & (1 << i):
                        trial = sol_mask & ~(1 << i)
                        covered = 0
                        for j in range(num_cand):
                            if trial & (1 << j):
                                covered |= cand_covers[j]
                        if covered == all_cliques_mask:
                            sol_mask = trial
                            improved = True
                            break
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Pair swap: remove one, add another
                for i in range(num_cand):
                    if not (sol_mask & (1 << i)):
                        continue
                    for j in range(num_cand):
                        if sol_mask & (1 << j):
                            continue
                        trial = (sol_mask & ~(1 << i)) | (1 << j)
                        covered = 0
                        for k in range(num_cand):
                            if trial & (1 << k):
                                covered |= cand_covers[k]
                        if covered == all_cliques_mask and bin(trial).count('1') < bin(sol_mask).count('1'):
                            sol_mask = trial
                            improved = True
                            break
                    if improved:
                        break
                if time.time() - start_time > DEADLINE - 10:
                    break
            return sol_mask
        
        best_mask = local_search(best_mask)
        best_size = bin(best_mask).count('1')
        
        best_vertices = [candidates[i] for i in range(num_cand) if best_mask & (1 << i)]
        return sorted(best_vertices)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)
        
    return results
