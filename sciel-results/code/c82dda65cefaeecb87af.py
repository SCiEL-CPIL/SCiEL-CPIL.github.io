import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 150s kill
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees of t-subsets (number of blocks covering each)
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]

    # Precompute scarcity weights for each block
    # weight[b] = sum of 1/degree[t] for t in block
    # This helps in the "Packing" phase to prefer blocks covering rare subsets
    block_scarcity_weights = [0.0] * num_k_subsets
    for b_idx, indices in enumerate(block_t_indices):
        w = 0.0
        for t_idx in indices:
            deg = t_subset_degrees[t_idx]
            if deg > 0:
                w += 1.0 / deg
        block_scarcity_weights[b_idx] = w

    # 2. Phase 1: Weighted Greedy Packing
    # Select blocks that cover many uncovered items and have high scarcity weight.
    # Conflict constraint: Do not select blocks that share any t-subset with already selected blocks?
    # No, that's too strict for a covering problem. We want a covering, not a packing.
    # However, the inspiration suggests "Greedy Packing" (maximal independent set heuristic) 
    # followed by repair. Let's stick to the inspiration's logic:
    # Select a core of blocks that are "non-conflicting" in terms of t-subsets?
    # If we strictly avoid sharing t-subsets, we are doing a packing, not a covering.
    # The inspiration code removes neighbors from the pool. This creates a set of blocks 
    # that are pairwise disjoint in terms of t-subsets. This is a very sparse core.
    # Then we repair the rest. This is a valid strategy.
    
    pool = set(range(num_k_subsets))
    active_blocks = set()
    cover_count = [0] * num_t_subsets
    uncovered_mask = [1] * num_t_subsets
    
    # We will run the packing phase. 
    # Score: Maximize (uncovered_count * 100 + scarcity_sum)
    # But we must remove conflicts to ensure the "packing" property.
    
    # Optimization: Instead of scanning the whole pool every time (O(N^2)), 
    # we can just scan. N is ~50k, iterations ~100-200. 10M ops. Acceptable.
    
    while pool:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        best_block = -1
        max_score = -1.0
        
        # To speed up, we might not scan the whole pool if it's huge, 
        # but 50k is small enough.
        for b_idx in pool:
            # Calculate score
            cov_uncovered = 0
            scarcity_sum = 0.0
            for t_idx in block_t_indices[b_idx]:
                if uncovered_mask[t_idx]:
                    cov_uncovered += 1
                    scarcity_sum += 1.0 / t_subset_degrees[t_idx]
            
            if cov_uncovered == 0:
                continue
                
            # Combine: Maximize covered count, tie-break by scarcity
            score = cov_uncovered * 100.0 + scarcity_sum
            
            if score > max_score:
                max_score = score
                best_block = b_idx
                
        if best_block == -1:
            break
            
        active_blocks.add(best_block)
        
        # Update cover_count and uncovered_mask
        for t_idx in block_t_indices[best_block]:
            cover_count[t_idx] += 1
            if cover_count[t_idx] == 1:
                uncovered_mask[t_idx] = 0
                
        # Remove best_block and its conflicts from pool
        # Conflicts: Any block sharing a t-subset with best_block
        neighbors_to_remove = set()
        for t_idx in block_t_indices[best_block]:
            neighbors_to_remove.update(t_subset_blocks[t_idx])
        
        pool.difference_update(neighbors_to_remove)
        
        if time.time() - start_time > TIME_LIMIT:
            break

    # 3. Phase 2: Residual Set Cover (Repair)
    # Now we have a core of non-conflicting blocks. 
    # We need to cover the remaining t-subsets.
    # Use standard greedy: Pick block covering most currently uncovered t-subsets.
    
    # Note: active_blocks contains the core. 
    # We should consider blocks NOT in active_blocks for repair.
    # Actually, we can consider any block, but if it's in active_blocks, 
    # its "gain" is 0 for the t-subsets it already covers (since they are covered).
    # But it might cover *other* uncovered t-subsets? No, if it was in the core, 
    # all its t-subsets were covered (uncovered_mask set to 0). 
    # Wait, the core blocks were selected based on uncovered_mask at the time.
    # So yes, any block in active_blocks covers only already covered t-subsets.
    # So we only need to look at blocks NOT in active_blocks.
    
    # Efficient repair:
    # Maintain a list of uncovered t-subsets.
    uncovered_t_list = [i for i in range(num_t_subsets) if cover_count[i] == 0]
    
    # To speed up, we can just iterate all k-subsets not in active_blocks.
    # But we need to know which ones cover uncovered items.
    
    while uncovered_t_list:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        best_block = -1
        max_gain = 0
        
        # Iterate all blocks
        for b_idx in range(num_k_subsets):
            if b_idx in active_blocks:
                continue
                
            gain = 0
            # Check how many uncovered t-subsets this block covers
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
                
        if best_block == -1 or max_gain == 0:
            break
            
        active_blocks.add(best_block)
        for t_idx in block_t_indices[best_block]:
            if cover_count[t_idx] == 0:
                cover_count[t_idx] = 1 # We just need to know it's covered
                # Remove from uncovered_t_list? 
                # It's easier to just check cover_count in the next loop.
                # But we need to update the list for efficiency if it gets small.
                # For now, we just rely on cover_count check in the gain calculation.
                # This is O(N * K * T) per iteration. 
                # N=50k, K=35, T=1. 1.75M ops per iteration. 
                # If we need 100 iterations, 175M ops. Might be slow in Python.
                
        # Optimization for Repair:
        # Instead of scanning all blocks every time, we can precompute 
        # which blocks cover which uncovered t-subsets?
        # The `t_subset_blocks` map helps.
        # Let's re-implement the repair loop using the map for speed.
        
        # Actually, the previous loop is fine if the number of iterations is small.
        # But let's make it safer.
        # If time is running out, break.

    # 4. Minimalization Helper
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    # Initial minimalization
    active_blocks, cover_count = minimize(active_blocks, cover_count)
    
    best_count = len(active_blocks)
    best_blocks = set(active_blocks)
    best_cover = list(cover_count)

    # 5. Local Search (Hill Climbing with Swaps)
    # This part is taken from the current program as it is efficient.
    while time.time() - start_time < TIME_LIMIT:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if time.time() - start_time > TIME_LIMIT:
                break
            if B not in active_blocks:
                continue
            
            # Find t-subsets uniquely covered by B
            uncovered_by_B = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    uncovered_by_B.append(t_idx)
            
            if not uncovered_by_B:
                continue

            # Sort by number of covering blocks to speed up intersection
            uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
            
            candidates = None
            for t_idx in uncovered_by_B:
                blocks_list = t_subset_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_list)
                else:
                    candidates.intersection_update(blocks_list)
                
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter candidates
            valid_candidates = []
            for c in candidates:
                if c != B and c not in active_blocks:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                continue
            
            # Perform Swap
            B_new = random.choice(valid_candidates)
            
            active_blocks.remove(B)
            active_blocks.add(B_new)
            
            for t_idx in block_t_indices[B]:
                cover_count[t_idx] -= 1
            for t_idx in block_t_indices[B_new]:
                cover_count[t_idx] += 1
            
            # Minimize after swap
            active_blocks, cover_count = minimize(active_blocks, cover_count)
            improved = True
            
            if len(active_blocks) < best_count:
                best_count = len(active_blocks)
                best_blocks = set(active_blocks)
                best_cover = list(cover_count)

        if not improved:
            # Try add-remove moves to escape local minima
            add_improved = False
            num_adds = 0
            max_adds = 5  # limit to keep time bounded
            
            for _ in range(max_adds):
                if time.time() - start_time > TIME_LIMIT:
                    break
                if num_adds >= max_adds:
                    break
                    
                # Pick a random uncovered t-subset to target
                # First find any t-subset with minimum coverage
                min_cov = min(cover_count)
                if min_cov >= 2:
                    break
                    
                target_t = random.choice([i for i, c in enumerate(cover_count) if c == min_cov])
                
                # Pick a random block that covers this target
                candidate_blocks = t_subset_blocks[target_t]
                if not candidate_blocks:
                    break
                    
                B_add = random.choice(candidate_blocks)
                if B_add in active_blocks:
                    continue
                    
                active_blocks.add(B_add)
                for t_idx in block_t_indices[B_add]:
                    cover_count[t_idx] += 1
                num_adds += 1
                add_improved = True
                
                # Now try to remove blocks
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                
                if len(active_blocks) < best_count:
                    best_count = len(active_blocks)
                    best_blocks = set(active_blocks)
                    best_cover = list(cover_count)
                    
                # Reset for next add
                if add_improved:
                    pass

            if not improved and not add_improved:
                break

    # Return best result found
    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result
