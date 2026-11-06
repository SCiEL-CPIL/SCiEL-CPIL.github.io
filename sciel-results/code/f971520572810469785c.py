import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Helper to generate Polarity Graph of PG(2, q)
    def get_polarity_graph(q):
        # Points are non-zero vectors in F_q^3, normalized
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = pow(x, -1, q)
                        points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                    elif y != 0:
                        inv_y = pow(y, -1, q)
                        points.append((0, 1, (z * inv_y) % q))
                    else:
                        inv_z = pow(z, -1, q)
                        points.append((0, 0, 1))
        
        unique_points = list(set(points))
        unique_points.sort()
        n_pts = len(unique_points)
        
        adj = np.zeros((n_pts, n_pts), dtype=np.int8)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                u = unique_points[i]
                v = unique_points[j]
                dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
                if dot == 0:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj, n_pts

    # Generate Paley graph P_47
    def get_paley_graph(n):
        # n must be a prime congruent to 1 mod 4 for Paley graph
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                # Check if d is a quadratic residue mod n
                is_residue = False
                for k in range(1, n):
                    if (k * k) % n == d:
                        is_residue = True
                        break
                if is_residue:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj

    adj_paley, n_paley = get_paley_graph(47)
    degs_paley = adj_paley.sum(axis=1)
    
    # Deterministic greedy algorithm on Paley graph
    def try_greedy_paley(n_target):
        if n_paley < n_target:
            return None
        
        target_deg = n_target - 39
        num_remove = n_paley - n_target
        
        # We need to select R of size num_remove such that for all v not in R:
        # deg_S(v) = deg_paley(v) - |N(v) ∩ R| >= target_deg
        # i.e., |N(v) ∩ R| <= deg_paley(v) - target_deg = limits[v]
        
        limits = degs_paley - target_deg
        
        # Greedy: iteratively pick vertex to remove that causes least constraint violation
        # Start with all vertices, remove num_remove vertices
        R = set()
        counts = np.zeros(n_paley, dtype=np.int8)  # counts[v] = |N(v) ∩ R|
        
        # Precompute neighbors for each vertex
        neighbors = [np.where(adj_paley[v] == 1)[0] for v in range(n_paley)]
        
        removed = 0
        while removed < num_remove:
            if time.time() > deadline:
                return None
            
            best_v = None
            best_score = -1
            best_violations = None
            
            for v in range(n_paley):
                if v in R:
                    continue
                
                # If we add v to R, what's the max violation?
                # For each neighbor u of v not in R: new_counts[u] = counts[u] + 1
                # Violation if new_counts[u] > limits[u]
                violations = 0
                for u in neighbors[v]:
                    if u not in R:
                        if counts[u] + 1 > limits[u]:
                            violations += 1
                
                # Score: prefer fewer violations, then prefer removing vertices with higher degree in S
                # deg_S(v) if not removed = degs_paley[v] - counts[v]
                score = -violations * 1000 + (degs_paley[v] - counts[v])
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_violations = violations
            
            if best_v is None:
                return None
            
            # Check if adding best_v causes any violation
            if best_violations > 0:
                # Try to find a vertex with 0 violations
                found_zero = False
                for v in range(n_paley):
                    if v in R:
                        continue
                    violations = 0
                    for u in neighbors[v]:
                        if u not in R:
                            if counts[u] + 1 > limits[u]:
                                violations += 1
                    if violations == 0:
                        best_v = v
                        found_zero = True
                        break
                if not found_zero:
                    # Can't make progress without violation, try random
                    candidates = [v for v in range(n_paley) if v not in R]
                    if not candidates:
                        return None
                    best_v = random.choice(candidates)
            
            R.add(best_v)
            removed += 1
            for u in neighbors[best_v]:
                counts[u] += 1
        
        # Verify the solution
        S = [i for i in range(n_paley) if i not in R]
        if len(S) != n_target:
            return None
        
        valid = True
        for v in S:
            if degs_paley[v] - counts[v] < target_deg:
                valid = False
                break
        
        if valid:
            edges = []
            old_to_new = {old: new for new, old in enumerate(S)}
            for new_i, old_i in enumerate(S):
                for new_j in range(new_i + 1, n_target):
                    old_j = S[new_j]
                    if adj_paley[old_i][old_j]:
                        edges.append([new_i, new_j])
            return {'n': n_target, 'edges': edges}
        
        return None

    # Try n from 47 down to 40 using greedy Paley
    for n in range(47, 39, -1):
        res = try_greedy_paley(n)
        if res:
            return res
    
    # Helper to generate Polarity Graph of PG(2, q) - keep as fallback
    adj_full, n_full = get_polarity_graph(7)
    degs_full = adj_full.sum(axis=1)
    
    def try_construct_from_subset(n_target, max_attempts=1000):
        if n_full < n_target:
            return None
        
        target_deg = n_target - 39
        num_remove = n_full - n_target
        
        limits = degs_full - target_deg
        
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
                
            R = set()
            counts = np.zeros(n_full, dtype=np.int8)
            
            candidates = list(range(n_full))
            random.shuffle(candidates)
            
            removed = 0
            valid_selection = True
            
            for v in candidates:
                if removed == num_remove:
                    break
                
                can_add = True
                neighbors = np.where(adj_full[v] == 1)[0]
                for u in neighbors:
                    if u not in R:
                        if counts[u] >= limits[u]:
                            can_add = False
                            break
                
                if can_add:
                    R.add(v)
                    removed += 1
                    for u in neighbors:
                        counts[u] += 1
            
            if removed == num_remove:
                S = [i for i in range(n_full) if i not in R]
                if len(S) != n_target:
                    continue
                
                valid = True
                for v in S:
                    if degs_full[v] - counts[v] < target_deg:
                        valid = False
                        break
                
                if valid:
                    edges = []
                    old_to_new = {old: new for new, old in enumerate(S)}
                    for new_i, old_i in enumerate(S):
                        for new_j in range(new_i + 1, n_target):
                            old_j = S[new_j]
                            if adj_full[old_i][old_j]:
                                edges.append([new_i, new_j])
                    return {'n': n_target, 'edges': edges}
        
        return None

    # Fallback: Try polarity graph subset
    for n in range(47, 39, -1):
        res = try_construct_from_subset(n)
        if res:
            return res
    
    # Fallback: Random Greedy Construction for lower n
    for n in range(40, 38, -1):
        if time.time() > deadline:
            break
        target_deg = n - 39
        adj = np.zeros((n, n), dtype=np.int8)
        degs = np.zeros(n, dtype=np.int8)
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            common = np.any(adj[u] & adj[v])
            if not common:
                adj[u][v] = 1
                adj[v][u] = 1
                degs[u] += 1
                degs[v] += 1
        
        if np.min(degs) >= target_deg:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}

    # Final fallback: n=39, empty graph
    return {'n': 39, 'edges': []}
