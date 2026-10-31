import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # Leave some buffer for the evaluator
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index for O(1) lookup
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets and their covered t-subsets
    # For v=19, k=7, t=4:
    # C(19,7) = 50388 blocks
    # Each block covers C(7,4) = 35 t-subsets
    # Total edges = 50388 * 35 = 1,763,580
    # This is manageable in memory and time.
    
    all_k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    # Reverse map: t_subset_index -> list of block indices
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for b_idx, block in enumerate(all_k_subsets):
        covered_ts = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            covered_ts.append(idx)
            ts_to_blocks[idx].append(b_idx)
        block_t_indices.append(covered_ts)
        
    def greedy_cover(randomize=True, top_k=1):
        """
        Greedy set cover construction.
        If randomize is True, pick randomly from the top 'top_k' candidates.
        """
        coverage = [0] * n_t_subsets
        selected = []
        selected_set = set()
        uncovered_count = n_t_subsets
        
        # Pre-compute "gain" for each block? No, gain changes as coverage increases.
        # We need to efficiently find the block with max gain.
        # For small N, we can just iterate.
        
        while uncovered_count > 0:
            best_gain = -1
            best_blocks = []
            
            # Iterate through all blocks to find best gain
            # Optimization: Only consider blocks that cover at least one uncovered subset?
            # Hard to filter dynamically without a heap. 
            # Given 50k blocks, iterating 150 times is 7.5M ops, which is fast in Python if optimized.
            
            for b_idx in range(n_k_subsets):
                if b_idx in selected_set:
                    continue
                
                gain = 0
                for ts_idx in block_t_indices[b_idx]:
                    if coverage[ts_idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [b_idx]
                elif gain == best_gain and gain > 0:
                    best_blocks.append(b_idx)
                    
            if not best_blocks:
                break # No more blocks can cover uncovered sets (should not happen)
                
            if randomize and len(best_blocks) > 1:
                # Pick randomly from top candidates
                # To add diversity, we can pick from a larger pool if we track top K
                # But finding top K requires sorting or a heap. 
                # For simplicity and speed, we just pick from the current best list.
                # If the best list is large, we can sample.
                if len(best_blocks) > top_k:
                    choice = random.choice(best_blocks[:top_k]) if top_k < len(best_blocks) else random.choice(best_blocks)
                else:
                    choice = random.choice(best_blocks)
            else:
                choice = best_blocks[0]
                
            selected.append(choice)
            selected_set.add(choice)
            
            for ts_idx in block_t_indices[choice]:
                if coverage[ts_idx] == 0:
                    uncovered_count -= 1
                coverage[ts_idx] += 1
                
        return selected

    def local_search(block_set):
        """
        Optimizes a valid covering by removing redundant blocks and performing swaps.
        """
        current_blocks = list(block_set)
        current_set = set(current_blocks)
        
        # Calculate coverage
        coverage = [0] * n_t_subsets
        for b_idx in current_blocks:
            for ts_idx in block_t_indices[b_idx]:
                coverage[ts_idx] += 1
        
        improved = True
        while improved and time.time() - start_time < time_limit * 0.95:
            improved = False
            
            # 1. Deletion Pass
            # Remove any block that does not uniquely cover any t-subset
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                can_remove = True
                for ts_idx in block_t_indices[b_idx]:
                    if coverage[ts_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for ts_idx in block_t_indices[b_idx]:
                        coverage[ts_idx] -= 1
                    current_blocks.pop(i)
                    current_set.discard(b_idx)
                    improved = True
                    break # Restart loop as indices shifted
            
            # 2. Swap Pass
            # Try to replace a block with another that covers all its unique subsets
            # and potentially allows further reductions.
            # Heuristic: Focus on blocks with many unique subsets, as they are "expensive" to replace.
            # Or, just iterate through all blocks.
            
            # To avoid O(N^2) complexity in the inner loop, we limit the search for swaps.
            # We only attempt swaps if we haven't improved via deletion.
            if not improved:
                # Shuffle to avoid bias
                indices = list(range(len(current_blocks)))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() - start_time > time_limit * 0.95:
                        break
                        
                    old_b = current_blocks[i]
                    unique_ts = [ts for ts in block_t_indices[old_b] if coverage[ts] == 1]
                    
                    if not unique_ts:
                        # This block is redundant, should have been caught by deletion?
                        # Maybe coverage changed? No, we just did deletion.
                        # If unique_ts is empty, it's redundant. Remove it.
                        for ts in block_t_indices[old_b]:
                            coverage[ts] -= 1
                        current_blocks.pop(i)
                        current_set.discard(old_b)
                        improved = True
                        break
                        
                    # Find candidate blocks that cover all unique_ts
                    # Start with the most constrained unique subset
                    unique_ts_sorted = sorted(unique_ts, key=lambda x: len(ts_to_blocks[x]))
                    
                    candidates = set(ts_to_blocks[unique_ts_sorted[0]])
                    for ts in unique_ts_sorted[1:]:
                        candidates.intersection_update(ts_to_blocks[ts])
                    
                    candidates.discard(old_b)
                    
                    if not candidates:
                        continue
                        
                    # Evaluate candidates
                    # Score: Number of other t-subsets covered by candidate that are currently covered by exactly 1 block (other than old_b)?
                    # Or simply: Maximize the number of t-subsets covered by candidate that are NOT unique to old_b.
                    # Actually, a good swap reduces the "tightness" of the cover.
                    # Let's just pick a random candidate if there are few, or the one covering most total subsets (which is constant k, so not useful).
                    # Let's pick the candidate that covers the most subsets that are currently covered by >1 blocks (increases slack).
                    
                    best_candidate = -1
                    best_score = -1
                    cand_list = list(candidates)
                    # Limit evaluation to a few candidates to save time
                    if len(cand_list) > 5:
                        cand_list = random.sample(cand_list, 5)
                        
                    for nb in cand_list:
                        score = 0
                        for ts in block_t_indices[nb]:
                            if coverage[ts] > 1:
                                score += 1
                        # Tie-breaker: random
                        if score > best_score or (score == best_score and random.random() < 0.5):
                            best_score = score
                            best_candidate = nb
                            
                    if best_candidate != -1:
                        # Perform swap
                        for ts in block_t_indices[old_b]:
                            coverage[ts] -= 1
                        for ts in block_t_indices[best_candidate]:
                            coverage[ts] += 1
                        current_blocks[i] = best_candidate
                        current_set.discard(old_b)
                        current_set.add(best_candidate)
                        improved = True
                        break
                        
        return current_blocks

    # Main Loop: Multiple Restarts
    best_solution = None
    best_size = float('inf')
    
    # Estimate time per iteration
    # Greedy is ~0.1s, Local Search is ~0.5s. Total ~0.6s.
    # We have 110s. So ~150 iterations?
    # Let's just run until time limit.
    
    while time.time() - start_time < time_limit * 0.9:
        # Greedy with randomization
        # Increase top_k occasionally to explore more?
        # Standard greedy is strong. Randomized greedy helps escape local minima.
        top_k = random.choice([1, 2, 5, 10])
        sol = greedy_cover(randomize=True, top_k=top_k)
        
        # Local Search
        optimized = local_search(sol)
        
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_solution = optimized
            
        # Early stop if we hit a very good solution?
        # Lower bound is 131. Best known 151.
        # If we find < 150, we are doing great.
        if best_size < 150:
             # We can try to squeeze more, but be careful with time.
             # Let's continue to see if we can go lower.
             pass

    if best_solution is None:
        # Fallback: Pure greedy
        sol = greedy_cover(randomize=False)
        best_solution = local_search(sol)
        
    return [list(all_k_subsets[i]) for i in best_solution]
