import time
import random
import numpy as np
import heapq

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

    # Pre-compute neighbor lists for faster access
    def compute_neighbors(adj, n):
        neighbors = []
        for i in range(n):
            neighbors.append(np.where(adj[i] == 1)[0])
        return neighbors

    adj_full, n_full = get_polarity_graph(7)
    # n_full should be 57
    # Check degrees
    degs_full = adj_full.sum(axis=1)
    neighbors_full = compute_neighbors(adj_full, n_full)
    
    def try_construct_from_subset(n_target, time_budget=10):
        if n_full < n_target:
            return None
        
        target_deg = n_target - 39
        num_remove = n_full - n_target
        
        # We need to select R of size num_remove
        # such that for all v not in R: |N(v) \cap R| <= degs_full[v] - target_deg
        limits = degs_full - target_deg
        
        # Strategy 1: Priority-based greedy (pick vertices with most slack)
        # Strategy 2: Random restarts
        # Strategy 3: Local search / hill climbing
        
        best_result = None
        attempt = 0
        t_start = time.time()
        
        while time.time() - t_start < time_budget and attempt < 5000:
            attempt += 1
            
            if attempt % 100 == 0 and time.time() > deadline:
                return best_result
            
            R = set()
            counts = np.zeros(n_full, dtype=np.int8)
            removed = 0
            
            # Priority queue: (slack, vertex) where slack = limits[v] - counts[v]
            # We want to pick vertices that are "safe" to remove
            # A vertex v is safe to add to R if for all u in N(v) \ R: counts[u] < limits[u]
            
            # Build initial priority based on limits
            # Vertices with high limits can absorb more removals from their neighbors
            # But we also want to pick vertices that have few neighbors already at limit
            
            if attempt <= 500:
                # Random shuffle approach
                candidates = list(range(n_full))
                random.shuffle(candidates)
                order = candidates
            else:
                # Priority-based: sort by limits descending (more slack first)
                # But also consider how many neighbors are near their limit
                scores = []
                for v in range(n_full):
                    # Score: higher is better to pick
                    # Base: limits[v] (more slack = better)
                    # Penalty: number of neighbors at or near limit
                    near_limit = 0
                    for u in neighbors_full[v]:
                        if counts[u] >= limits[u] - 1:
                            near_limit += 1
                    score = limits[v] * 10 - near_limit
                    scores.append((-score, v))  # Min-heap, so negate
                order = [v for _, v in sorted(scores)]
            
            for v in order:
                if removed == num_remove:
                    break
                if v in R:
                    continue
                
                # Check if v can be added to R
                can_add = True
                for u in neighbors_full[v]:
                    if u not in R:
                        if counts[u] >= limits[u]:
                            can_add = False
                            break
                
                if can_add:
                    R.add(v)
                    removed += 1
                    for u in neighbors_full[v]:
                        counts[u] += 1
            
            if removed == num_remove:
                # Verify the solution
                S = [i for i in range(n_full) if i not in R]
                if len(S) != n_target:
                    continue
                
                # Check min degree in S
                valid = True
                for v in S:
                    if degs_full[v] - counts[v] < target_deg:
                        valid = False
                        break
                
                if valid:
                    # Construct the edge list for S
                    edges = []
                    for new_i, old_i in enumerate(S):
                        for new_j in range(new_i + 1, n_target):
                            old_j = S[new_j]
                            if adj_full[old_i][old_j]:
                                edges.append([new_i, new_j])
                    return {'n': n_target, 'edges': edges}
            
            # Local search: if we got close, try to repair
            if removed >= num_remove - 3:
                # Try to swap some vertices in R to fix violations
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    
                    # Find vertices in S that violate
                    violators = []
                    for v in range(n_full):
                        if v not in R and degs_full[v] - counts[v] < target_deg:
                            violators.append(v)
                    
                    if not violators:
                        break
                    
                    # Try to move a violator out of S (into R) and move someone else in
                    v_viol = random.choice(violators)
                    # Find a vertex in R that can be moved to S
                    candidates_swap = []
                    for u in R:
                        # Check if removing u from R (adding to S) is safe
                        # After swap: v_viol goes to R, u goes to S
                        # For w in N(v_viol) \ R: counts[w] + 1 <= limits[w]
                        # For w in N(u) \ (R \ {u}): counts[w] - 1 >= 0 (always true)
                        # For w in S \ {u}: need degs_full[w] - (counts[w] - 1) >= target_deg
                        #   i.e., counts[w] - 1 <= degs_full[w] - target_deg = limits[w]
                        #   which is always true since counts[w] <= limits[w]
                        # Main constraint: for w in N(v_viol) \ R, counts[w] + 1 <= limits[w]
                        
                        can_swap = True
                        for w in neighbors_full[v_viol]:
                            if w not in R:
                                if counts[w] + 1 > limits[w]:
                                    can_swap = False
                                    break
                        
                        if can_swap:
                            candidates_swap.append(u)
                    
                    if candidates_swap:
                        u_swap = random.choice(candidates_swap)
                        # Perform swap
                        R.remove(u_swap)
                        R.add(v_viol)
                        # Update counts
                        for w in neighbors_full[v_viol]:
                            counts[w] += 1
                        for w in neighbors_full[u_swap]:
                            counts[w] -= 1
            
            if time.time() > deadline:
                break
        
        return best_result

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        # Allocate more time for larger n
        time_budget = min(remaining_time * 0.8, 30)
        res = try_construct_from_subset(n, time_budget)
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
            # Check C4
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
