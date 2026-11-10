import time
import random

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
        
        # Greedy initialization
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
        
        initial_solution = greedy_cover()
        best_size = len(initial_solution)
        best_solution = list(initial_solution)
        
        # Stronger lower bound using greedy maximal packing
        def lower_bound(uncovered_mask):
            remaining = uncovered_mask
            lb = 0
            while remaining:
                # Find clique with smallest "weight" (fewest uncovered cliques intersecting it)
                # Actually, pick a random uncovered clique and greedily pack
                # Better: pick the clique that intersects the fewest other uncovered cliques
                best_c = -1
                best_intersect = m + 1
                temp = remaining
                while temp:
                    ci = (temp & -temp).bit_length() - 1
                    # Count how many uncovered cliques intersect clique ci
                    intersect_count = 0
                    temp2 = remaining
                    while temp2:
                        cj = (temp2 & -temp2).bit_length() - 1
                        if clique_masks[ci] & clique_masks[cj]:
                            intersect_count += 1
                        temp2 &= temp2 - 1
                    if intersect_count < best_intersect:
                        best_intersect = intersect_count
                        best_c = ci
                    temp &= temp - 1
                
                if best_c == -1:
                    break
                    
                lb += 1
                # Remove all cliques intersecting best_c
                intersect_mask = 0
                temp = remaining
                while temp:
                    cj = (temp & -temp).bit_length() - 1
                    if clique_masks[best_c] & clique_masks[cj]:
                        intersect_mask |= (1 << cj)
                    temp &= temp - 1
                remaining &= ~intersect_mask
            return lb
        
        # Branch and bound with bitmask state
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
            if current_size + lower_bound(uncovered_mask) >= best_size:
                return
            
            # Pick the most constrained uncovered clique
            # Find uncovered clique with fewest vertices
            best_c = -1
            best_len = n + 1
            temp = uncovered_mask
            while temp:
                ci = (temp & -temp).bit_length() - 1
                clen = bin(clique_masks[ci]).count('1')
                if clen < best_len:
                    best_len = clen
                    best_c = ci
                temp &= temp - 1
            
            if best_c == -1:
                return
            
            # Get vertices in this clique
            target_mask = clique_masks[best_c]
            vertices = []
            temp = target_mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                vertices.append(v)
                temp &= temp - 1
            
            # Sort by coverage (descending)
            vertices.sort(key=lambda v: bin(uncovered_mask & vertex_covers[v]).count('1'), reverse=True)
            
            for v in vertices:
                new_uncovered = uncovered_mask & ~vertex_covers[v]
                current_solution.append(v)
                branch_and_bound(new_uncovered, current_size + 1, current_solution, depth + 1)
                current_solution.pop()
                
                if time.time() > deadline:
                    return
        
        # Start B&B
        branch_and_bound(all_cliques_mask, 0, [])
        
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
                    for i in range(len(best_solution)):
                        for v in range(n):
                            if v in best_solution:
                                continue
                            candidate = best_solution[:i] + best_solution[i+1:]
                            candidate.append(v)
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
