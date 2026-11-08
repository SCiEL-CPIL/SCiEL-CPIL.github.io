# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    def generate_candidates(max_weight=4):
        """Generate all nonzero integer vectors with entries in {-1, 0, 1} up to given weight."""
        candidates = []
        # Weight 2
        for i, j in combinations(range(D), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
        # Weight 3
        if max_weight >= 3:
            for i, j, k in combinations(range(D), 3):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
        # Weight 4
        if max_weight >= 4:
            for i, j, k, l in combinations(range(D), 4):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            for sl in [1, -1]:
                                v = [0] * D
                                v[i] = si
                                v[j] = sj
                                v[k] = sk
                                v[l] = sl
                                candidates.append(v)
        return np.array(candidates, dtype=np.int64)

    def vectorized_greedy(candidates, threshold):
        """
        Fast greedy independent set using NumPy.
        Conflict if dist2 < threshold.
        """
        n = len(candidates)
        if n == 0:
            return np.array([], dtype=int)
        
        norms2 = np.sum(candidates * candidates, axis=1)
        dot = candidates @ candidates.T
        dist2 = norms2[:, None] + norms2[None, :] - 2 * dot
        
        # Conflict matrix: True if dist2 < threshold
        conflict = (dist2 < threshold)
        np.fill_diagonal(conflict, False)
        
        selected = []
        remaining = np.ones(n, dtype=bool)
        
        while np.any(remaining):
            # Count conflicts within remaining
            # degrees[i] = number of j in remaining such that conflict[i,j] is True
            # We can compute this by multiplying conflict (restricted to remaining) with ones
            # But conflict is n x n. 
            # Let's just sum rows of conflict where cols are remaining.
            # conflict[remaining, :] is a submatrix.
            # Sum over columns for each row in remaining.
            
            # Optimization: Only compute degrees for remaining nodes
            # This is still O(N^2) per step in worst case, but NumPy is fast.
            
            # Get indices of remaining
            rem_idx = np.where(remaining)[0]
            if len(rem_idx) == 0: break
            
            # Calculate degrees for remaining nodes
            # conflict[rem_idx, :] has shape (len(rem_idx), n)
            # We want sum over j in rem_idx.
            # Mask out non-remaining columns?
            # Actually, conflict[i,j] is symmetric.
            # deg_i = sum_{j in remaining} conflict[i,j]
            
            # Efficient way:
            # Create a mask for remaining
            # Sum the conflict matrix rows for remaining nodes, but only for remaining columns.
            
            # Since n can be up to 5280 (weight 4), n^2 is ~27M. 
            # Storing conflict matrix is 27M bytes (bool). OK.
            # Summing 5280 x 5280 matrix 5280 times is too slow in Python loop.
            
            # Alternative: 
            # We can update the remaining set by removing the chosen node and its neighbors.
            # This is standard greedy.
            
            # To speed up, we can use a heuristic: pick a random node? Or min degree?
            # Min degree requires O(N) work per step.
            
            # Let's try a simple random greedy first to see if it beats 220.
            # Or better: 
            # If we just want to beat 220, maybe weight-4 is not the way.
            
            # Let's stick to the baseline 220 if greedy is too slow or doesn't improve.
            
            # For now, return empty if we can't do it fast.
            # But we can do a few steps.
            
            break # Placeholder, we will replace with actual logic or fallback
            
        return np.array(selected, dtype=int)

    # Strategy 1: Baseline weight-2 (220 vectors)
    candidates_w2 = generate_candidates(max_weight=2)
    best_result = candidates_w2
    best_k = len(best_result)
    
    # Strategy 2: Try weight-4 with vectorized greedy
    # Only if we have time.
    if time.time() < deadline - 10:
        candidates_w4 = generate_candidates(max_weight=4)
        # Note: generate_candidates(max_weight=4) includes w2, w3, w4.
        # We want ONLY w4 to avoid mixing norms.
        # If we mix, max norm is 2 (from w4). Threshold 4.
        # w2 vectors have norm sqrt(2). w4 have norm 2.
        # Dist between w2 and w4?
        # If w2 is sub of w4, dist^2 = 4 (from the other 2 non-zeros of w4? No).
        # u=(1,1,0,0), v=(1,1,1,1). v-u=(0,0,1,1). Dist^2=2.
        # 2 < 4. Conflict.
        # So we cannot mix w2 and w4 with threshold 4.
        # We must use homogeneous sets.
        
        # Extract only weight-4 candidates
        # We can filter candidates_w4 by norm squared == 4
        norms2_w4 = np.sum(candidates_w4 * candidates_w4, axis=1)
        w4_mask = (norms2_w4 == 4)
        w4_cands = candidates_w4[w4_mask]
        
        # Run greedy on w4_cands
        # We need an efficient greedy.
        # Given the time limit, a simple randomized greedy might work better than min-degree if min-degree is slow.
        
        n = len(w4_cands)
        if n > 0:
            norms2 = np.sum(w4_cands * w4_cands, axis=1)
            dot = w4_cands @ w4_cands.T
            dist2 = norms2[:, None] + norms2[None, :] - 2 * dot
            conflict = (dist2 < 4)
            np.fill_diagonal(conflict, False)
            
            remaining = np.ones(n, dtype=bool)
            selected_indices = []
            
            # Randomized greedy with degree check
            # To be safe, let's just return the baseline if this is too complex.
            # But let's try a simple pass.
            
            # Actually, the number of weight-4 vectors is 5280.
            # The independence number of the graph where edges connect vectors with dist^2 < 4.
            # Conflict if they share 3 indices with matching signs.
            # This is a sparse graph.
            
            # Let's implement a fast min-degree greedy using lists.
            # But 5280 nodes, O(N^2) init is fine.
            # Greedy steps: N steps. Each step O(N) to find min deg, O(N) to remove.
            # Total O(N^2) = 27M operations. Python might take 10-20s.
            # NumPy can do the degree update faster.
            
            # Let's try.
            deg = np.sum(conflict, axis=1)
            
            while np.any(remaining):
                # Find min degree among remaining
                # Set deg of non-remaining to inf
                deg_temp = deg.copy()
                deg_temp[~remaining] = np.inf
                min_idx = np.argmin(deg_temp)
                
                if not remaining[min_idx]:
                    break
                    
                selected_indices.append(min_idx)
                
                # Remove min_idx and its neighbors
                neighbors = np.where(conflict[min_idx])[0]
                remaining[neighbors] = False
                remaining[min_idx] = False
                
                # Update degrees for remaining nodes
                # For each remaining node u, if it was connected to min_idx, its degree decreases by 1.
                # conflict[min_idx] is the mask of neighbors.
                # We need to decrement deg for all u in remaining such that conflict[min_idx, u] is True.
                # This is a vectorized operation.
                deg[remaining & conflict[min_idx]] -= 1
                
                if time.time() > deadline - 5:
                    break
            
            w4_result = w4_cands[selected_indices]
            if len(w4_result) > best_k:
                best_result = w4_result
                best_k = len(w4_result)

    return best_result
# EVOLVE-BLOCK-END
