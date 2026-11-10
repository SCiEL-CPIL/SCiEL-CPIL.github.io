import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 18
    
    def is_k4_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                        continue
                    for l in range(k + 1, n):
                        if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                            return False
        return True

    def get_triangles(adj):
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                k_mask = common
                while k_mask:
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    if k > j:
                        triangles.append((i, j, k))
                    k_mask ^= lsb
        return triangles

    def max_triangle_free_size(adj, triangles):
        if not triangles:
            return n
        
        tri_masks = [0] * len(triangles)
        for idx, (i, j, k) in enumerate(triangles):
            tri_masks[idx] = (1 << i) | (1 << j) | (1 << k)
            
        for k in range(n, 0, -1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                contains_tri = False
                for tm in tri_masks:
                    if (mask & tm) == tm:
                        contains_tri = True
                        break
                if not contains_tri:
                    return k
        return 0

    def count_k4_violations(adj):
        """Count number of K4 cliques in the graph."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                        continue
                    for l in range(k + 1, n):
                        if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                            count += 1
        return count

    def evaluate_clauses(adj, target_size=12):
        """
        Evaluate clause satisfaction:
        - K4 clauses: each K4 must be broken (unsatisfied if K4 exists)
        - Triangle clauses: every subset of size target_size must contain a triangle
          (unsatisfied if a subset of size target_size is triangle-free)
        
        Returns (k4_count, missing_tri_count) where missing_tri_count is the number
        of target_size-subsets that are triangle-free.
        """
        # K4 violations
        k4_count = count_k4_violations(adj)
        
        # For triangle-free subsets of size target_size:
        # A subset S is triangle-free if no triangle is fully contained in S.
        # Count how many subsets of size target_size are triangle-free.
        triangles = get_triangles(adj)
        if not triangles:
            # Empty graph: all C(18,12) subsets are triangle-free
            missing_tri = 18564  # C(18,12)
            return k4_count, missing_tri
        
        tri_masks = [(1 << i) | (1 << j) | (1 << k) for (i, j, k) in triangles]
        
        # For each subset of size target_size, check if it contains any triangle
        # This is expensive, so use a smarter approach:
        # A vertex set S is triangle-free iff for every triangle (a,b,c), not all of a,b,c are in S.
        # Equivalently, S is a subset of the complement of at least one triangle's vertex set? No.
        # S is triangle-free iff S does NOT contain any triangle entirely.
        
        # Use inclusion or just iterate subsets
        # C(18,12) = C(18,6) = 18564, manageable
        missing_tri = 0
        for subset in itertools.combinations(range(n), target_size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            has_tri = False
            for tm in tri_masks:
                if (mask & tm) == tm:
                    has_tri = True
                    break
            if not has_tri:
                missing_tri += 1
        
        return k4_count, missing_tri

    def min_sat_local_search(adj, max_iter=5000):
        """
        Local search on edge flips to minimize clause violations.
        Objective: minimize (k4_count * weight_k4 + missing_tri_count * weight_tri)
        where weight_k4 is high to prioritize K4-freeness.
        """
        weight_k4 = 1000
        weight_tri = 1
        
        def objective(a):
            k4, mt = evaluate_clauses(a, 12)
            return k4 * weight_k4 + mt
        
        current_obj = objective(adj)
        best_obj = current_obj
        best_adj = adj[:]
        
        # Precompute all edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try flipping random edges
            improved = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                edge = random.choice(all_edges)
                i, j = edge
                # Flip edge
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                
                new_obj = objective(adj)
                
                if new_obj < current_obj:
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_adj = adj[:]
                    improved = True
                else:
                    # Revert
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
            
            if not improved:
                # Try multiple flips simultaneously (simulated annealing style)
                accepted = False
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    e1 = random.choice(all_edges)
                    e2 = random.choice(all_edges)
                    if e1 == e2:
                        continue
                    # Flip both
                    for (i, j) in [e1, e2]:
                        adj[i] ^= (1 << j)
                        adj[j] ^= (1 << i)
                    
                    new_obj = objective(adj)
                    
                    # Accept if better or with small probability if worse (SA-like)
                    if new_obj <= current_obj or random.random() < 0.1:
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_adj = adj[:]
                        accepted = True
                    else:
                        # Revert
                        for (i, j) in [e1, e2]:
                            adj[i] ^= (1 << j)
                            adj[j] ^= (1 << i)
                
                if not accepted:
                    # Reset to best
                    adj = best_adj[:]
                    current_obj = best_obj
        
        return best_adj

    best_score = 12
    best_edges = None
    
    # Seed construction: K6,6,6 (complete tripartite)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    seed_adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    seed_adj[a] |= (1 << b)
                    seed_adj[b] |= (1 << a)
    
    # Verify seed
    seed_triangles = get_triangles(seed_adj)
    seed_s = max_triangle_free_size(seed_adj, seed_triangles)
    best_score = seed_s
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if seed_adj[i] & (1 << j):
                best_edges.append([i, j])
    
    # Min-SAT local search from seed
    search_adj = seed_adj[:]
    improved_adj = min_sat_local_search(search_adj, max_iter=3000)
    
    if is_k4_free(improved_adj):
        tri = get_triangles(improved_adj)
        s = max_triangle_free_size(improved_adj, tri)
        if s < best_score:
            best_score = s
            best_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if improved_adj[i] & (1 << j):
                        best_edges.append([i, j])
    
    # Try more local search iterations with different random seeds
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Start from best known or a perturbation
        if random.random() < 0.5:
            start_adj = best_edges and [0] * n or seed_adj[:]
            # Reconstruct from best_edges if available
            if best_edges:
                start_adj = [0] * n
                for e in best_edges:
                    start_adj[e[0]] |= (1 << e[1])
                    start_adj[e[1]] |= (1 << e[0])
            else:
                start_adj = seed_adj[:]
        else:
            # Random perturbation of best
            start_adj = [0] * n
            if best_edges:
                for e in best_edges:
                    start_adj[e[0]] |= (1 << e[1])
                    start_adj[e[1]] |= (1 << e[0])
            else:
                start_adj = seed_adj[:]
            # Flip a few random edges
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            for _ in range(random.randint(1, 4)):
                i, j = random.choice(all_edges)
                start_adj[i] ^= (1 << j)
                start_adj[j] ^= (1 << i)
        
        # Run local search
        result_adj = min_sat_local_search(start_adj, max_iter=1000)
        
        if is_k4_free(result_adj):
            tri = get_triangles(result_adj)
            s = max_triangle_free_size(result_adj, tri)
            if s < best_score:
                best_score = s
                best_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if result_adj[i] & (1 << j):
                            best_edges.append([i, j])
        
        if best_score <= 10:
            break
    
    return best_edges
