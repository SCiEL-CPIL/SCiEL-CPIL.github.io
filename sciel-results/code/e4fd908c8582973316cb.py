<<<<<<< SEARCH
    def local_search(initial_pts, max_time=250):
        pts = list(initial_pts)
        improved = True
        max_swap_attempts = 100  # Limit swap attempts to avoid getting stuck
        search_start = time.time()
        
        while improved and time.time() - search_start < max_time:
            improved = False
            
            # Phase 1: Try adding any valid point
            # Generate all candidates
            candidates = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
            
            # Filter out points already in pts
            pts_set = set(pts)
            mask_not_in = np.array([tuple(c) not in pts_set for c in candidates], dtype=bool)
            candidates = candidates[mask_not_in]
            
            if len(candidates) > 0:
                # Precompute D_sq for current points
                P_arr = np.array(pts, dtype=np.int64)
                D_sq = compute_D_sq(P_arr)
                
                valid_mask = is_valid_addition_batch(pts, candidates, D_sq)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # Pick a random valid one to add diversity
                    idx = valid_indices[random.randint(0, len(valid_indices)-1)]
                    p = tuple(candidates[idx])
                    pts.append(p)
                    improved = True
            
            # Phase 2: If no point could be added, try swaps
            if not improved:
                # Try swapping a random subset of points
                swap_candidates = random.sample(range(len(pts)), min(max_swap_attempts, len(pts)))
                for i in swap_candidates:
                    if time.time() - search_start > max_time - 5:
                        return pts
                    old_pt = pts[i]
                    pts_temp = pts[:i] + pts[i+1:]
                    
                    # Candidates for replacement
                    candidates = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
                    pts_temp_set = set(pts_temp)
                    mask_not_in = np.array([tuple(c) not in pts_temp_set for c in candidates], dtype=bool)
                    candidates = candidates[mask_not_in]
                    
                    if len(candidates) == 0:
                        continue
                        
                    # Precompute D_sq for temp points
                    P_temp_arr = np.array(pts_temp, dtype=np.int64)
                    D_sq_temp = compute_D_sq(P_temp_arr)
                    
                    valid_mask = is_valid_addition_batch(pts_temp, candidates, D_sq_temp)
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # Pick a random valid replacement to avoid cycles
                        idx = valid_indices[random.randint(0, len(valid_indices)-1)]
                        p = tuple(candidates[idx])
                        pts = pts_temp + [p]
                        improved = True
                        break
                        
        return pts
=======
    def count_violations(P):
        """
        Count the number of isosceles triangles in the set of points P.
        P: numpy array of shape (N, 2)
        Returns: integer count of isosceles triangles.
        """
        n = len(P)
        if n < 3:
            return 0
        
        violations = 0
        # For each point i, check distances to all other points
        # If distance d appears k times, it forms k*(k-1)/2 isosceles triangles with apex i
        for i in range(n):
            # Compute squared distances from P[i] to all other points
            dists = np.sum((P - P[i])**2, axis=1)
            # Exclude distance to self (which is 0)
            dists = np.delete(dists, i)
            
            if len(dists) < 2:
                continue
                
            # Count frequencies
            unique_dists, counts = np.unique(dists, return_counts=True)
            # For each distance, add C(count, 2)
            violations += np.sum(counts * (counts - 1) // 2)
            
        return violations

    def simulated_annealing(initial_pts, max_time=60):
        """
        Perform simulated annealing to maximize the number of points
        while minimizing isosceles triangles.
        """
        if not initial_pts:
            return []
            
        P = np.array(initial_pts, dtype=np.int64)
        current_violations = count_violations(P)
        best_P = P.copy()
        best_count = len(P)
        
        start_time = time.time()
        T = 10.0
        cooling = 0.995
        
        all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
        
        while time.time() - start_time < max_time and T > 0.1:
            # Random move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_P = P.copy()
            
            if move_type == 'add':
                idx = random.randint(0, len(all_points)-1)
                p = all_points[idx]
                # Check if already in P
                if not any(np.all(P[i] == p) for i in range(len(P))):
                    new_P = np.vstack([P, p])
                else:
                    continue
                    
            elif move_type == 'remove':
                if len(P) > 1:
                    idx = random.randint(0, len(P)-1)
                    new_P = np.delete(P, idx, axis=0)
                else:
                    continue
                    
            elif move_type == 'swap':
                if len(P) > 0:
                    idx_remove = random.randint(0, len(P)-1)
                    new_P = np.delete(P, idx_remove, axis=0)
                    
                    idx_add = random.randint(0, len(all_points)-1)
                    p_add = all_points[idx_add]
                    # Check if p_add is in new_P
                    if not any(np.all(new_P[i] == p_add) for i in range(len(new_P))):
                        new_P = np.vstack([new_P, p_add])
                    else:
                        new_P = P.copy()
                        continue
                        
            # Evaluate new state
            new_violations = count_violations(new_P)
            
            # Objective: maximize count, minimize violations.
            # Score = count * 100 - violations
            old_score = len(P) * 100 - current_violations
            new_score = len(new_P) * 100 - new_violations
            delta_score = new_score - old_score
            
            if delta_score > 0:
                accept = True
            else:
                accept_prob = np.exp(delta_score / T)
                accept = random.random() < accept_prob
                
            if accept:
                P = new_P
                current_violations = new_violations
                
                # Update best ONLY if valid
                if current_violations == 0 and len(P) > best_count:
                    best_P = P.copy()
                    best_count = len(P)
                    
            T *= cooling
            
        return list(map(tuple, best_P))
>>>>>>> REPLACE

Now update the main loop to use `simulated_annealing`.

