import time
import random
import math
import sys
sys.setrecursionlimit(10000)

def construct():
    """
    Finds minimum vertex transversals for maximal cliques (size >= 2) 
    in three specific random graphs.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Graph Generation
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    # 2. Bron-Kerbosch for Maximal Cliques
    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P union X to minimize recursion
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

    # 3. Set Cover Solver using bitmasks for speed
    def solve_transversal(cliques, n):
        if not cliques:
            return []
        
        m = len(cliques)
        
        # Precompute: for each clique, bitmask of vertices
        clique_masks = []
        for c in cliques:
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks.append(mask)
        
        # For each vertex, bitmask of cliques it covers
        vertex_covers = [0] * n
        for i, cm in enumerate(clique_masks):
            temp = cm
            while temp:
                v = (temp & -temp).bit_length() - 1
                vertex_covers[v] |= (1 << i)
                temp &= temp - 1
        
        all_cliques_mask = (1 << m) - 1
        
        # Precompute pairwise clique intersection masks for faster lower bound
        # For each clique, which other cliques share at least one vertex
        clique_intersect = [0] * m
        for i in range(m):
            temp = all_cliques_mask
            while temp:
                j = (temp & -temp).bit_length() - 1
                if i != j and (clique_masks[i] & clique_masks[j]):
                    clique_intersect[i] |= (1 << j)
                temp &= temp - 1
        
        # Lagrangian/subgradient-guided greedy initialization
        def lagrangian_greedy():
            uncovered = all_cliques_mask
            solution = []
            
            # Initialize dual weights for each clique
            weights = [1.0] * m
            
            # Subgradient iteration to adjust weights
            for iteration in range(50):
                if time.time() > deadline:
                    break
                    
                # Select vertex with best ratio: sum(weights of covered cliques) / 1
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
                
                # Update weights: increase weights for uncovered cliques
                # Subgradient step: for each uncovered clique, increase its weight
                # based on how "hard" it is to cover
                for ci in range(m):
                    if (1 << ci) & new_uncovered:
                        # Count how many remaining vertices can cover this clique
                        coverable = 0
                        for v in range(n):
                            if v not in solution and (vertex_covers[v] & (1 << ci)):
                                coverable += 1
                        # Harder cliques (fewer covering vertices) get higher weight
                        if coverable > 0:
                            weights[ci] += 1.0 / coverable
                        else:
                            weights[ci] += 10.0  # Very hard to cover
                    else:
                        # Slightly decay weights for covered cliques
                        weights[ci] *= 0.95
                
                uncovered = new_uncovered
                if not uncovered:
                    break
            
            # Finish with standard greedy if needed
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
                if time.time() > deadline:
                    break
            
            return solution
        
        # Also keep simple greedy as fallback
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
                if time.time() > deadline:
                    return solution
            return solution
        
        # Try both and pick the better one
        sol1 = lagrangian_greedy()
        sol2 = greedy_cover()
        
        if len(sol1) <= len(sol2):
            initial_solution = sol1
        else:
            initial_solution = sol2
        
        best_size = len(initial_solution)
        best_solution = list(initial_solution)
        
        # Stronger lower bound using greedy maximal packing with precomputed intersections
        def lower_bound(uncovered_mask):
            remaining = uncovered_mask
            lb = 0
            while remaining:
                # Pick the uncovered clique that intersects the fewest other uncovered cliques
                best_c = -1
                best_intersect = m + 1
                temp = remaining
                while temp:
                    ci = (temp & -temp).bit_length() - 1
                    # Count uncovered cliques that intersect clique ci
                    intersect_mask = clique_intersect[ci] & remaining
                    intersect_count = bin(intersect_mask).count('1')
                    if intersect_count < best_intersect:
                        best_intersect = intersect_count
                        best_c = ci
                    temp &= temp - 1
                
                if best_c == -1:
                    break
                    
                lb += 1
                # Remove all uncovered cliques intersecting best_c (including itself)
                intersect_mask = (clique_intersect[best_c] | (1 << best_c)) & remaining
                remaining &= ~intersect_mask
            return lb
        
        # Even stronger: LP-style lower bound using fractional covering
        def lower_bound_lp(uncovered_mask):
            """Fractional lower bound: sum of 1/|N(C)| for each uncovered clique C,
            where N(C) is the set of vertices that can cover C and are still available."""
            lb = 0.0
            temp = uncovered_mask
            while temp:
                ci = (temp & -temp).bit_length() - 1
                # Count vertices that can cover clique ci
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
            return math.ceil(lb * 0.99)  # Slight adjustment for integrality
        
        # Branch and bound with bitmask state - improved with better pruning
        def branch_and_bound(uncovered_mask, current_size, current_solution, depth=0):
            nonlocal best_size, best_solution
            
            if time.time() > deadline:
                return
                
            if uncovered_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_solution)
                return
                
            # Pruning with lower bound
            lb = lower_bound(uncovered_mask)
            if current_size + lb >= best_size:
                return
            
            # Additional LP-based pruning
            lb_lp = lower_bound_lp(uncovered_mask)
            if current_size + lb_lp >= best_size:
                return
            
            # Pick the most constrained uncovered clique
            # Find uncovered clique with fewest vertices, tie-break by fewest covering vertices
            best_c = -1
            best_len = n + 1
            best_cover_count = n + 1
            temp = uncovered_mask
            while temp:
                ci = (temp & -temp).bit_length() - 1
                clen = bin(clique_masks[ci]).count('1')
                # Count vertices that can cover this clique
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
            
            # Get vertices in this clique that can actually cover it
            target_mask = clique_masks[best_c]
            vertices = []
            temp = target_mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                if vertex_covers[v] & (1 << best_c):
                    vertices.append(v)
                temp &= temp - 1
            
            # Sort by coverage (descending), with tie-breaking by vertex degree
            vertices.sort(key=lambda v: (bin(uncovered_mask & vertex_covers[v]).count('1'), -len(adj[v])), reverse=True)
            
            for v in vertices:
                new_uncovered = uncovered_mask & ~vertex_covers[v]
                current_solution.append(v)
                branch_and_bound(new_uncovered, current_size + 1, current_solution, depth + 1)
                current_solution.pop()
                
                if time.time() > deadline:
                    return
        
        # Start B&B
        branch_and_bound(all_cliques_mask, 0, [])
        
        # Randomized restarts: try multiple randomized greedy initializations
        if time.time() < deadline - 10:
            rng = random.Random(42)
            for restart in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Randomized greedy with perturbation
                uncovered = all_cliques_mask
                sol = []
                while uncovered:
                    # Collect candidate vertices with their scores
                    candidates = []
                    for v in range(n):
                        if v in sol:
                            continue
                        count = bin(uncovered & vertex_covers[v]).count('1')
                        if count > 0:
                            # Add random perturbation
                            score = count + rng.random() * 0.5
                            candidates.append((score, v))
                    
                    if not candidates:
                        break
                    
                    # Pick from top candidates with some randomness
                    candidates.sort(reverse=True)
                    top_k = min(3, len(candidates))
                    chosen = candidates[rng.randint(0, top_k - 1)][1]
                    sol.append(chosen)
                    uncovered &= ~vertex_covers[chosen]
                
                if len(sol) < best_size:
                    # Verify and update
                    covered = 0
                    for v in sol:
                        covered |= vertex_covers[v]
                    if covered == all_cliques_mask:
                        best_size = len(sol)
                        best_solution = list(sol)
        
        # Local search: try to improve by removing vertices and re-adding optimally
        if best_size > 0 and time.time() < deadline - 5:
            # Try to remove each vertex from best_solution and see if we can cover with fewer
            improved = True
            while improved:
                improved = False
                current = list(best_solution)
                for i in range(len(current)):
                    # Try removing current[i]
                    candidate = current[:i] + current[i+1:]
                    # Check if candidate still covers all cliques
                    covered = 0
                    for v in candidate:
                        covered |= vertex_covers[v]
                    if covered == all_cliques_mask:
                        # Valid improvement
                        best_solution = candidate
                        best_size = len(candidate)
                        improved = True
                        break
                # Also try swapping: remove one vertex, add another
                if not improved and len(best_solution) > 0:
                    # Precompute coverage for current solution minus each vertex
                    for i in range(len(best_solution)):
                        base_covered = 0
                        for j, v in enumerate(best_solution):
                            if j != i:
                                base_covered |= vertex_covers[v]
                        
                        # Find uncovered cliques
                        uncovered = all_cliques_mask & ~base_covered
                        if not uncovered:
                            # Can remove this vertex
                            best_solution = best_solution[:i] + best_solution[i+1:]
                            best_size = len(best_solution)
                            improved = True
                            break
                        
                        # Try to find a single vertex that covers all uncovered
                        best_swap_v = -1
                        best_swap_score = -1
                        for v in range(n):
                            if v in best_solution:
                                continue
                            cover_mask = vertex_covers[v]
                            # Count how many uncovered cliques this vertex covers
                            covered_count = bin(cover_mask & uncovered).count('1')
                            if covered_count == bin(uncovered).count('1'):
                                # This vertex covers all uncovered cliques
                                best_swap_v = v
                                best_swap_score = covered_count
                                break
                            elif covered_count > best_swap_score:
                                best_swap_score = covered_count
                                best_swap_v = v
                        
                        if best_swap_v != -1:
                            candidate = best_solution[:i] + best_solution[i+1:] + [best_swap_v]
                            covered = 0
                            for x in candidate:
                                covered |= vertex_covers[x]
                            if covered == all_cliques_mask and len(candidate) < best_size:
                                best_solution = candidate
                                best_size = len(candidate)
                                improved = True
                                break
                    if improved:
                        break
                
                # Try 2-exchange: remove two vertices, add two others
                if not improved and len(best_solution) >= 2 and time.time() < deadline - 10:
                    found = False
                    for i in range(len(best_solution)):
                        if found:
                            break
                        for j in range(i + 1, len(best_solution)):
                            if found:
                                break
                            # Compute coverage without vertices i and j
                            base_covered = 0
                            for k, v in enumerate(best_solution):
                                if k != i and k != j:
                                    base_covered |= vertex_covers[v]
                            
                            uncovered = all_cliques_mask & ~base_covered
                            if not uncovered:
                                # Can remove both
                                candidate = best_solution[:i] + best_solution[i+1:j] + best_solution[j+1:]
                                best_solution = candidate
                                best_size = len(candidate)
                                improved = True
                                found = True
                                break
                            
                            # Try to find two vertices that cover all uncovered
                            uncovered_count = bin(uncovered).count('1')
                            best_pair = None
                            best_pair_score = -1
                            
                            # Try pairs of vertices
                            for v1 in range(n):
                                if v1 in best_solution:
                                    continue
                                cover1 = vertex_covers[v1]
                                count1 = bin(cover1 & uncovered).count('1')
                                if count1 == 0:
                                    continue
                                if count1 == uncovered_count:
                                    best_pair = (v1, -1)
                                    best_pair_score = count1
                                    break
                                for v2 in range(v1 + 1, n):
                                    if v2 in best_solution:
                                        continue
                                    cover2 = vertex_covers[v2]
                                    count2 = bin((cover1 | cover2) & uncovered).count('1')
                                    if count2 == uncovered_count:
                                        best_pair = (v1, v2)
                                        best_pair_score = count2
                                        break
                                    elif count2 > best_pair_score:
                                        best_pair_score = count2
                                        best_pair = (v1, v2)
                                if best_pair and best_pair_score == uncovered_count:
                                    break
                            
                            if best_pair and best_pair_score == uncovered_count:
                                v1, v2 = best_pair
                                candidate = best_solution[:i] + best_solution[i+1:j] + best_solution[j+1:]
                                candidate.append(v1)
                                if v2 != -1:
                                    candidate.append(v2)
                                covered = 0
                                for x in candidate:
                                    covered |= vertex_covers[x]
                                if covered == all_cliques_mask and len(candidate) < best_size:
                                    best_solution = candidate
                                    best_size = len(candidate)
                                    improved = True
                                    found = True
                                    break
                    if improved:
                        break
                
                # Run B&B from the current solution to see if we can do better
                if not improved and time.time() < deadline - 15:
                    # Try B&B starting from current solution, looking for smaller
                    current_covered = 0
                    for v in best_solution:
                        current_covered |= vertex_covers[v]
                    remaining_uncovered = all_cliques_mask & ~current_covered
                    if remaining_uncovered == 0:
                        # Already valid, try to find smaller via B&B with tighter bound
                        # Set best_size to current and run B&B to find smaller
                        old_best_size = best_size
                        old_best_solution = list(best_solution)
                        best_size = old_best_size  # Keep same, B&B will try to improve
                        branch_and_bound(all_cliques_mask, 0, [])
                        if best_size >= old_best_size:
                            best_size = old_best_size
                            best_solution = old_best_solution
        
        return best_solution

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Allocate time proportionally based on graph size
    total_weight = sum(n**1.5 for n, _, _ in instances)
    time_budget = deadline - start_time - 5  # Reserve 5s for final processing
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        if remaining < 2:
            # Not enough time, use greedy fallback
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u, v)}
            results.append(sorted(touched))
            continue
        
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            results.append([])
            continue
        
        solution = solve_transversal(cliques, n)
        results.append(sorted(solution))

    return results
