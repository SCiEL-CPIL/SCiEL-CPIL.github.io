import time
import random
import math
import sys
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        U = P.union(X)
        if not U:
            u = None
        else:
            u = max(U, key=lambda x: len(adj[x].intersection(P)))
        
        for v in list(P.difference(adj[u])):
            bron_kerbosch(adj, R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]), cliques)
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        cliques = []
        P = set(range(n))
        X = set()
        bron_kerbosch(adj, set(), P, X, cliques)
        return cliques

    def solve_transversal(cliques, n, time_limit):
        if not cliques:
            return []
        
        m = len(cliques)
        
        clique_masks = []
        for c in cliques:
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks.append(mask)
        
        vertex_covers = [0] * n
        for i, cm in enumerate(clique_masks):
            temp = cm
            while temp:
                v = (temp & -temp).bit_length() - 1
                vertex_covers[v] |= (1 << i)
                temp &= temp - 1
        
        all_cliques_mask = (1 << m) - 1
        
        # Precompute for faster lower bound
        clique_intersect = [0] * m
        for i in range(m):
            temp = all_cliques_mask
            while temp:
                j = (temp & -temp).bit_length() - 1
                if i != j and (clique_masks[i] & clique_masks[j]):
                    clique_intersect[i] |= (1 << j)
                temp &= temp - 1
        
        def lower_bound(uncovered_mask):
            remaining = uncovered_mask
            lb = 0
            while remaining:
                best_c = -1
                best_intersect = m + 1
                temp = remaining
                while temp:
                    ci = (temp & -temp).bit_length() - 1
                    intersect_mask = clique_intersect[ci] & remaining
                    intersect_count = bin(intersect_mask).count('1')
                    if intersect_count < best_intersect:
                        best_intersect = intersect_count
                        best_c = ci
                    temp &= temp - 1
                if best_c == -1:
                    break
                lb += 1
                intersect_mask = (clique_intersect[best_c] | (1 << best_c)) & remaining
                remaining &= ~intersect_mask
            return lb

        def lower_bound_lp(uncovered_mask):
            lb = 0.0
            temp = uncovered_mask
            while temp:
                ci = (temp & -temp).bit_length() - 1
                target = clique_masks[ci]
                count = 0
                t = target
                while t:
                    v = (t & -t).bit_length() - 1
                    if vertex_covers[v] & (1 << ci):
                        count += 1
                    t &= t - 1
                if count > 0:
                    lb += 1.0 / count
                temp &= temp - 1
            return math.ceil(lb * 0.99)

        def greedy_cover():
            uncovered = all_cliques_mask
            solution = []
            while uncovered:
                best_v = -1
                best_count = 0
                for v in range(n):
                    count = bin(uncovered & vertex_covers[v]).count('1')
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_count == 0:
                    break
                solution.append(best_v)
                uncovered &= ~vertex_covers[best_v]
            return solution

        # Lagrangian greedy
        def lagrangian_greedy():
            uncovered = all_cliques_mask
            solution = []
            weights = [1.0] * m
            
            for iteration in range(30):
                if time.time() > time_limit:
                    break
                best_v = -1
                best_score = -1
                for v in range(n):
                    cover_mask = vertex_covers[v]
                    if not (cover_mask & uncovered):
                        continue
                    score = 0.0
                    temp = cover_mask & uncovered
                    while temp:
                        ci = (temp & -temp).bit_length() - 1
                        score += weights[ci]
                        temp &= temp - 1
                    if score > best_score:
                        best_score = score
                        best_v = v
                
                if best_v == -1:
                    break
                    
                solution.append(best_v)
                new_uncovered = uncovered & ~vertex_covers[best_v]
                
                for ci in range(m):
                    if (1 << ci) & new_uncovered:
                        coverable = 0
                        for v in range(n):
                            if v not in solution and (vertex_covers[v] & (1 << ci)):
                                coverable += 1
                        if coverable > 0:
                            weights[ci] += 1.0 / coverable
                        else:
                            weights[ci] += 10.0
                    else:
                        weights[ci] *= 0.95
                
                uncovered = new_uncovered
                if not uncovered:
                    break
            
            while uncovered:
                best_v = -1
                best_count = 0
                for v in range(n):
                    if v in solution:
                        continue
                    count = bin(uncovered & vertex_covers[v]).count('1')
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_count == 0:
                    break
                solution.append(best_v)
                uncovered &= ~vertex_covers[best_v]
            
            return solution

        sol1 = lagrangian_greedy()
        sol2 = greedy_cover()
        initial_solution = sol1 if len(sol1) <= len(sol2) else sol2
        best_size = len(initial_solution)
        best_solution = list(initial_solution)

        def branch_and_bound(uncovered_mask, current_size, current_solution):
            nonlocal best_size, best_solution
            
            if time.time() > time_limit:
                return
                
            if uncovered_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_solution)
                return
                
            lb = lower_bound(uncovered_mask)
            if current_size + lb >= best_size:
                return
            
            lb_lp = lower_bound_lp(uncovered_mask)
            if current_size + lb_lp >= best_size:
                return
            
            best_c = -1
            best_len = n + 1
            best_cover_count = n + 1
            temp = uncovered_mask
            while temp:
                ci = (temp & -temp).bit_length() - 1
                clen = bin(clique_masks[ci]).count('1')
                target = clique_masks[ci]
                cover_count = 0
                t = target
                while t:
                    v = (t & -t).bit_length() - 1
                    if vertex_covers[v] & (1 << ci):
                        cover_count += 1
                    t &= t - 1
                if clen < best_len or (clen == best_len and cover_count < best_cover_count):
                    best_len = clen
                    best_cover_count = cover_count
                    best_c = ci
                temp &= temp - 1
            
            if best_c == -1:
                return
            
            target_mask = clique_masks[best_c]
            vertices = []
            temp = target_mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                if vertex_covers[v] & (1 << best_c):
                    vertices.append(v)
                temp &= temp - 1
            
            vertices.sort(key=lambda v: bin(uncovered_mask & vertex_covers[v]).count('1'), reverse=True)
            
            for v in vertices:
                new_uncovered = uncovered_mask & ~vertex_covers[v]
                current_solution.append(v)
                branch_and_bound(new_uncovered, current_size + 1, current_solution)
                current_solution.pop()
                if time.time() > time_limit:
                    return

        branch_and_bound(all_cliques_mask, 0, [])

        # Local search with efficient coverage tracking
        if best_size > 0 and time.time() < time_limit - 2:
            # Precompute coverage of current solution
            def get_coverage(sol):
                covered = 0
                for v in sol:
                    covered |= vertex_covers[v]
                return covered

            improved = True
            while improved and time.time() < time_limit - 2:
                improved = False
                current = list(best_solution)
                
                # Try removing each vertex
                for i in range(len(current)):
                    candidate = current[:i] + current[i+1:]
                    covered = get_coverage(candidate)
                    if covered == all_cliques_mask:
                        best_solution = candidate
                        best_size = len(candidate)
                        improved = True
                        break
                
                # Try 1-exchange: remove one, add one
                if not improved and len(best_solution) > 0:
                    for i in range(len(best_solution)):
                        base_covered = 0
                        for j, v in enumerate(best_solution):
                            if j != i:
                                base_covered |= vertex_covers[v]
                        
                        uncovered = all_cliques_mask & ~base_covered
                        if not uncovered:
                            best_solution = best_solution[:i] + best_solution[i+1:]
                            best_size = len(best_solution)
                            improved = True
                            break
                        
                        # Find a vertex that covers all uncovered
                        for v in range(n):
                            if v in best_solution:
                                continue
                            if (vertex_covers[v] & uncovered) == uncovered:
                                candidate = best_solution[:i] + best_solution[i+1:] + [v]
                                if len(candidate) < best_size:
                                    best_solution = candidate
                                    best_size = len(candidate)
                                    improved = True
                                    break
                        if improved:
                            break
                
                # Try 2-exchange: remove two, add two
                if not improved and len(best_solution) >= 2 and time.time() < time_limit - 5:
                    found = False
                    for i in range(len(best_solution)):
                        if found:
                            break
                        for j in range(i + 1, len(best_solution)):
                            if found:
                                break
                            base_covered = 0
                            for k, v in enumerate(best_solution):
                                if k != i and k != j:
                                    base_covered |= vertex_covers[v]
                            
                            uncovered = all_cliques_mask & ~base_covered
                            if not uncovered:
                                candidate = best_solution[:i] + best_solution[i+1:j] + best_solution[j+1:]
                                best_solution = candidate
                                best_size = len(candidate)
                                improved = True
                                found = True
                                break
                            
                            # Try to find two vertices covering all uncovered
                            uncovered_count = bin(uncovered).count('1')
                            best_pair = None
                            for v1 in range(n):
                                if v1 in best_solution:
                                    continue
                                cover1 = vertex_covers[v1]
                                if (cover1 & uncovered) == uncovered:
                                    best_pair = (v1, -1)
                                    break
                                for v2 in range(v1 + 1, n):
                                    if v2 in best_solution:
                                        continue
                                    cover2 = vertex_covers[v2]
                                    if ((cover1 | cover2) & uncovered) == uncovered:
                                        best_pair = (v1, v2)
                                        break
                                    if best_pair is None:
                                        best_pair = (v1, v2)
                                if best_pair and best_pair[1] != -1:
                                    # Check if this pair covers all
                                    c1, c2 = best_pair
                                    if ((vertex_covers[c1] | vertex_covers[c2]) & uncovered) == uncovered:
                                        break
                                    else:
                                        best_pair = None
                                else:
                                    break
                            
                            if best_pair:
                                v1, v2 = best_pair
                                candidate = best_solution[:i] + best_solution[i+1:j] + best_solution[j+1:]
                                candidate.append(v1)
                                if v2 != -1:
                                    candidate.append(v2)
                                if len(candidate) < best_size:
                                    best_solution = candidate
                                    best_size = len(candidate)
                                    improved = True
                                    found = True
                                    break
                    if improved:
                        break

        return best_solution

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Allocate time: give more time to larger graphs
    time_budgets = [35.0, 45.0, 55.0]  # Total 135s, leaving 10s margin
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        if remaining < 3:
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u, v)}
            results.append(sorted(touched))
            continue
        
        instance_time_limit = min(time_budgets[idx], remaining - 2.0)
        
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            results.append([])
            continue
        
        solution = solve_transversal(cliques, n, time.time() + instance_time_limit)
        results.append(sorted(solution))

    return results
