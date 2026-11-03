import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Map each k-block to the list of t-subsets it contains
    # We store these as numpy arrays for faster indexing in local search
    subsets_of_block = [None] * n_k
    # Map each t-subset to the list of k-blocks that contain it
    blocks_of_subset = [[] for _ in range(n_t)]
    
    # Build adjacency
    for j, B in enumerate(k_subsets):
        subs = []
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subs.append(i)
            blocks_of_subset[i].append(j)
        subsets_of_block[j] = np.array(subs, dtype=np.int32)
        
    # Convert blocks_of_subset to numpy arrays as well for faster intersection checks if needed,
    # but lists are fine for membership if we convert to sets on the fly or use np.isin.
    # Actually, for local search 2-swap, we need to check if a block covers a set of critical subsets.
    # Let's keep blocks_of_subset as lists for now, but we might need sets.
    blocks_of_subset_sets = [set(lst) for lst in blocks_of_subset]

    best_solution = None
    best_size = float('inf')

    # Helper to check if a solution is valid
    def is_valid(sol):
        if not sol:
            return n_t == 0
        covered = np.zeros(n_t, dtype=bool)
        for b_idx in sol:
            covered[subsets_of_block[b_idx]] = True
        return np.all(covered)

    def calc_cover_counts(sol):
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in sol:
            counts[subsets_of_block[b_idx]] += 1
        return counts

    def local_search(sol):
        if not sol:
            return sol
        sol = list(sol)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # 1. Deletion
            cover_count = calc_cover_counts(sol)
            new_sol = []
            for b_idx in sol:
                subs = subsets_of_block[b_idx]
                if np.all(cover_count[subs] > 1):
                    cover_count[subs] -= 1
                    changed = True
                else:
                    new_sol.append(b_idx)
            sol = new_sol
            
            # 2. 2-Swap
            # Try to replace two blocks with one
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                cover_count = calc_cover_counts(sol)
                sol_set = set(sol)
                n_sol = len(sol)
                
                for i in range(n_sol):
                    if time.time() > deadline - 5:
                        break
                    b1 = sol[i]
                    for j in range(i + 1, n_sol):
                        if time.time() > deadline - 5:
                            break
                        b2 = sol[j]
                        
                        # Find critical subsets: those with count 1 or 2 that become 0 if b1, b2 removed
                        critical = []
                        
                        # Check subsets of b1
                        for s in subsets_of_block[b1]:
                            c = cover_count[s]
                            if c == 1:
                                critical.append(s)
                            elif c == 2:
                                # If s is in b2, removing both makes it 0.
                                if b2 in blocks_of_subset_sets[s]:
                                    critical.append(s)
                        
                        # Check subsets of b2 that are not in b1
                        # We only need to check if they become 0.
                        # If s in b2, not in b1.
                        # If c==1, removing b2 makes it 0.
                        # If c==2, removing b2 leaves 1 (since not in b1). So not critical.
                        for s in subsets_of_block[b2]:
                            if b1 not in blocks_of_subset_sets[s]:
                                if cover_count[s] == 1:
                                    critical.append(s)
                        
                        if not critical:
                            # Remove both
                            new_sol = [b for idx, b in enumerate(sol) if idx not in (i, j)]
                            sol = new_sol
                            improved = True
                            break
                        
                        # Try to find a block k covering all critical subsets
                        if critical:
                            # Use the smallest list to start intersection
                            min_len = float('inf')
                            min_s = -1
                            for s in critical:
                                if len(blocks_of_subset_sets[s]) < min_len:
                                    min_len = len(blocks_of_subset_sets[s])
                                    min_s = s
                            
                            candidates = blocks_of_subset_sets[min_s].copy()
                            for s in critical:
                                if s != min_s:
                                    candidates &= blocks_of_subset_sets[s]
                                    if not candidates:
                                        break
                            
                            if candidates:
                                # Filter out b1, b2 and existing blocks
                                candidates.discard(b1)
                                candidates.discard(b2)
                                candidates -= sol_set
                                
                                if candidates:
                                    # Pick a candidate
                                    k = next(iter(candidates))
                                    new_sol = [b for idx, b in enumerate(sol) if idx not in (i, j)]
                                    new_sol.append(k)
                                    sol = new_sol
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    # Restart deletion
                    changed = True

        return sol

    # 3. Greedy Construction with Restarts
    num_restarts = 50 # Increase restarts since each is faster
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize weights: Random uniform is often good for diversity
        block_weights = np.random.uniform(0, 1, n_k)
        
        # Greedy Selection
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        # Use a simple loop to find max weight block covering new elements
        # Optimization: Precompute incidence? No, n_k * n_t is large.
        # We will iterate through blocks and check coverage.
        
        while not np.all(covered):
            if time.time() > deadline - 10:
                break
            
            best_b = -1
            max_score = -1
            
            # To speed up, we can maintain a list of active blocks
            # But for simplicity, we iterate all unpicked blocks
            # Optimization: Only consider blocks that cover at least one uncovered subset
            # We can precompute for each block which subsets it covers.
            
            # Vectorized scoring:
            # For each block j, score = sum(block_weights[j] * (1 if subset s is uncovered and in block j))
            # This is hard to vectorize across blocks easily without sparse matrix.
            
            # Let's stick to a simple loop but break early if possible?
            # Actually, let's use the incidence structure.
            # For each uncovered subset s, the blocks covering it are blocks_of_subset[s].
            # We want to pick a block j that maximizes sum of weights of new subsets covered.
            
            # Alternative: Randomized Greedy
            # Pick a random uncovered subset s.
            # Among blocks covering s, pick the one with max "score" (weight * coverage of other uncovered subsets).
            
            if not np.any(~covered):
                break
                
            uncovered_indices = np.where(~covered)[0]
            # Pick a random uncovered subset
            s_rand = uncovered_indices[np.random.randint(len(uncovered_indices))]
            
            candidates = blocks_of_subset[s_rand]
            
            best_candidate = -1
            max_candidate_score = -1
            
            for b_idx in candidates:
                if picked[b_idx]:
                    continue
                # Calculate score: how many NEW subsets does this block cover?
                # And weighted by block_weights?
                # Simple heuristic: Just count new subsets.
                # Or use block_weights as a tie-breaker or multiplier.
                
                # Let's use: score = sum(block_weights[b_idx] for new subsets)
                # Actually, just count new subsets is standard greedy.
                # Let's mix: score = count_new + 0.1 * block_weights[b_idx]
                
                new_count = 0
                subs = subsets_of_block[b_idx]
                for s in subs:
                    if not covered[s]:
                        new_count += 1
                
                score = new_count + 0.01 * block_weights[b_idx]
                
                if score > max_candidate_score:
                    max_candidate_score = score
                    best_candidate = b_idx
            
            if best_candidate == -1:
                break # No valid block found? Should not happen if problem is coverable.
            
            picked[best_candidate] = True
            solution_blocks.append(best_candidate)
            
            # Update covered
            covered[subsets_of_block[best_candidate]] = True
            
            # Update weights? 
            # Standard greedy doesn't update weights, it just picks best coverage.
            # Weighted greedy updates weights.
            # Let's just use standard greedy with random restarts.
            
        if np.all(covered) and len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            local_sol = local_search(solution_blocks)
            if len(local_sol) < best_size:
                best_size = len(local_sol)
                best_solution = local_sol

    # Fallback if no solution found
    if best_solution is None:
        best_solution = []
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        while not np.all(covered):
            best_b = -1
            max_new = -1
            for j in range(n_k):
                if not picked[j]:
                    new_cov = np.sum(~covered[subsets_of_block[j]])
                    if new_cov > max_new:
                        max_new = new_cov
                        best_b = j
            if best_b == -1:
                break
            best_solution.append(best_b)
            picked[best_b] = True
            covered[subsets_of_block[best_b]] = True
        best_size = len(best_solution)
        local_sol = local_search(best_solution)
        if len(local_sol) < best_size:
            best_size = len(local_sol)
            best_solution = local_sol

    return [list(k_subsets[idx]) for idx in best_solution]
