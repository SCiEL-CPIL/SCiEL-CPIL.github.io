"""Covering design C(v,k,t) via GRASP with bitmask greedy + enhanced local search."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin under 150s limit

    # 1. Precomputation
    # Generate all k-subsets (blocks)
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    n_k = len(all_blocks)

    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)

    # For each block, compute the bitmask of t-subsets it covers
    # Using Python integers as bitsets (arbitrary precision)
    block_masks = [0] * n_k
    # Also store the list of subset indices for each block for faster local search
    block_subsets = [[] for _ in range(n_k)]
    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_index[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs

    all_mask = (1 << num_subsets) - 1

    # Precompute for each t-subset, which blocks cover it (for local search swaps)
    # t_to_blocks[t_idx] = list of block indices
    t_to_blocks = [[] for _ in range(num_subsets)]
    for i in range(n_k):
        for s_idx in block_subsets[i]:
            t_to_blocks[s_idx].append(i)
    # Convert to sets for faster intersection in local search
    t_to_blocks_sets = [set(lst) for lst in t_to_blocks]

    def greedy_construct(alpha=0.0, rng=None):
        """
        Greedy set cover construction.
        If alpha == 0, standard greedy (max gain).
        If alpha > 0, GRASP-like: choose from RCL of size proportional to alpha.
        Returns list of block indices.
        """
        if rng is None:
            rng = random.Random(42)
        
        covered = 0
        selected_indices = []
        
        while covered != all_mask:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gain for all blocks
            gains = np.empty(n_k, dtype=np.int32)
            # This loop is the bottleneck. We can optimize by only checking if gain > 0.
            # But we need the max for RCL.
            # Let's use a list for gains to avoid numpy overhead in the tight loop if n_k is small enough?
            # n_k is ~75k. NumPy might be faster for the vectorized bit_count if we could, but we can't easily vectorize bit_count on a list of ints.
            # So we stick to Python loop but optimize it.
            
            best_gain = 0
            best_idx = -1
            # For GRASP, we need the top gains.
            # Let's just find the max gain first, then filter.
            
            for i in range(n_k):
                gain = (block_masks[i] & (all_mask ^ covered)).bit_count()
                gains[i] = gain
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_gain == 0:
                break
            
            if alpha == 0.0:
                # Standard greedy
                selected_indices.append(best_idx)
                covered |= block_masks[best_idx]
            else:
                # GRASP: Create RCL
                # Threshold: gain >= best_gain * alpha? Or gain >= best_gain - alpha * range?
                # Let's use: gain >= best_gain * alpha is too restrictive if alpha < 1.
                # Standard GRASP: RCL contains elements with score >= alpha * best + (1-alpha) * worst?
                # Or simply: gain >= best_gain * alpha.
                # Let's try: gain >= best_gain * alpha.
                threshold = best_gain * alpha
                candidates = []
                for i in range(n_k):
                    if gains[i] >= threshold and gains[i] > 0:
                        candidates.append(i)
                
                if not candidates:
                    candidates = [best_idx]
                
                # Pick randomly from candidates
                # To keep it fast, don't shuffle huge lists. Just pick one.
                # But random.choice is O(1) if we have the list.
                # Building the list is O(n_k).
                # We can optimize by breaking early if we have enough candidates?
                # No, we need the full list for uniform random choice.
                # However, if alpha is close to 1, the list is small.
                # If alpha is small, the list is large.
                
                # Optimization: If the list is very large, just pick the best one?
                # That defeats the purpose.
                # Let's limit the RCL size to, say, 50 best candidates.
                # Find top 50 gains.
                # This is O(n_k * 50) if we do it naively.
                # Better: Just collect all candidates, and if len > 50, sample 50.
                
                if len(candidates) > 50:
                    # Sample 50
                    candidates = rng.sample(candidates, 50)
                
                chosen = rng.choice(candidates)
                selected_indices.append(chosen)
                covered |= block_masks[chosen]
        
        return selected_indices

    def local_search(selected_indices, rng):
        """
        Enhanced local search: redundancy removal, swaps, 2-for-1.
        Returns a new list of selected indices.
        """
        if time.time() > deadline - 5.0:
            return selected_indices
            
        selected_set = set(selected_indices)
        remaining = list(selected_indices)
        
        # Compute initial counts
        subset_counts = np.zeros(num_subsets, dtype=np.int32)
        for idx in selected_set:
            for s_idx in block_subsets[idx]:
                subset_counts[s_idx] += 1

        improved = True
        max_iterations = 100 # Safety cap on iterations
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            iterations = 0
            while iterations < max_iterations:
                iterations += 1
                if time.time() > deadline - 5.0:
                    break
                
                # 1. Remove redundant blocks
                changed = False
                for i in range(len(remaining)):
                    bi = remaining[i]
                    ts = block_subsets[bi]
                    # Check if all counts > 1
                    # Using numpy for speed
                    if np.all(subset_counts[ts] > 1):
                        subset_counts[ts] -= 1
                        remaining.pop(i)
                        selected_set.remove(bi)
                        changed = True
                        improved = True
                        break # Restart loop after change
                
                if changed:
                    continue
                
                # 2. Try swaps
                # Find a block that has unique coverage (count == 1)
                # Try to replace it with another block that covers those unique subsets
                # and minimizes new waste.
                
                # To speed up, sample a few blocks to try swapping
                if len(remaining) > 10:
                    # Sample up to 10 blocks to try swapping
                    sample_size = min(10, len(remaining))
                    indices_to_try = rng.sample(range(len(remaining)), sample_size)
                else:
                    indices_to_try = list(range(len(remaining)))
                
                swap_found = False
                for i in indices_to_try:
                    if i >= len(remaining):
                        continue
                    bi = remaining[i]
                    ts = block_subsets[bi]
                    
                    # Find unique t-subsets for this block
                    # unique_ts = ts[subset_counts[ts] == 1]
                    unique_mask = (subset_counts[ts] == 1)
                    if not np.any(unique_mask):
                        # This block is fully redundant? Should have been caught above.
                        # But maybe counts changed?
                        subset_counts[ts] -= 1
                        remaining.pop(i)
                        selected_set.remove(bi)
                        improved = True
                        swap_found = True
                        break
                    
                    # ts is a list, unique_mask is a numpy array.
                    # Cannot index list with numpy array directly.
                    unique_ts = [x for x, m in zip(ts, unique_mask) if m]
                    
                    # Find candidates that cover all unique_ts
                    # Intersection of t_to_blocks_sets for each unique_ts
                    if len(unique_ts) == 0:
                        continue
                        
                    candidates = set(t_to_blocks_sets[unique_ts[0]])
                    for ut in unique_ts[1:]:
                        candidates.intersection_update(t_to_blocks_sets[ut])
                    
                    # Remove blocks already in the solution
                    candidates.difference_update(selected_set)
                    
                    if not candidates:
                        continue
                    
                    # Evaluate candidates
                    # We want to minimize the "cost" of the new block.
                    # Cost = number of subsets in the new block that are already covered (waste)
                    # Or simply: prefer blocks that cover more unique subsets? 
                    # Actually, we MUST cover the unique subsets.
                    # So we look at the rest of the block.
                    
                    best_candidate = None
                    best_cost = float('inf')
                    
                    cand_list = list(candidates)
                    # Limit candidates to check
                    if len(cand_list) > 20:
                        cand_list = rng.sample(cand_list, 20)
                        
                    for new_bi in cand_list:
                        new_ts = block_subsets[new_bi]
                        # Cost: how many of new_ts are already covered (count > 0)?
                        # Actually, we remove bi, so counts for bi's ts drop by 1.
                        # So for new_bi, we check:
                        # For each s in new_ts:
                        #   current_count = subset_counts[s]
                        #   if s is in bi's ts: effective_count = current_count - 1
                        #   else: effective_count = current_count
                        #   if effective_count == 0: we gain coverage (good)
                        #   else: we add waste (bad)
                        
                        # Simulate:
                        # waste = 0
                        # for s in new_ts:
                        #    cnt = subset_counts[s]
                        #    if s in bi_ts_set: cnt -= 1
                        #    if cnt > 0: waste += 1
                        
                        # To make this fast, we can precompute a set of bi's ts?
                        # bi_ts_set = set(ts)
                        # This is O(k choose t) per swap attempt.
                        
                        # Let's approximate cost as: number of s in new_ts where subset_counts[s] > 1
                        # This ignores the -1 from removing bi.
                        # If s is in bi, count drops by 1. If count was 1, it becomes 0 (good).
                        # If count was > 1, it stays > 0 (waste).
                        # If s is not in bi, count stays same. If count > 0 (waste), if count == 0 (gain).
                        
                        # Let's just count how many s in new_ts have subset_counts[s] > 1.
                        # This is a proxy for waste.
                        waste = np.sum(subset_counts[new_ts] > 1)
                        
                        if waste < best_cost:
                            best_cost = waste
                            best_candidate = new_bi
                            
                    if best_candidate is not None:
                        # Perform swap
                        # Remove bi
                        subset_counts[ts] -= 1
                        remaining.pop(i)
                        selected_set.remove(bi)
                        
                        # Add new_bi
                        new_ts = block_subsets[best_candidate]
                        subset_counts[new_ts] += 1
                        remaining.append(best_candidate)
                        selected_set.add(best_candidate)
                        
                        improved = True
                        swap_found = True
                        break # Restart local search loop
                        
                if swap_found:
                    continue
                    
                # 3. Try 2-for-1
                # Replace two blocks with one if possible
                if len(remaining) >= 2:
                    # Sample a few pairs
                    pairs_to_try = min(5, len(remaining) * (len(remaining) - 1) // 2)
                    for _ in range(pairs_to_try):
                        if time.time() > deadline - 5.0:
                            break
                        i1, i2 = rng.sample(range(len(remaining)), 2)
                        b1, b2 = remaining[i1], remaining[i2]
                        ts1 = block_subsets[b1]
                        ts2 = block_subsets[b2]
                        
                        # Find t-subsets uniquely covered by b1 or b2
                        # unique_to_b1 = ts1 where count == 1 AND not in ts2
                        # unique_to_b2 = ts2 where count == 1 AND not in ts1
                        # shared_unique = ts in both where count == 1
                        
                        # Efficient way:
                        # Create a mask for ts1 and ts2?
                        # Or just iterate.
                        
                        # Let's gather all unique subsets
                        unique_set = set()
                        
                        # For ts1
                        for s in ts1:
                            if subset_counts[s] == 1:
                                # Check if in ts2
                                if s not in ts2: # O(1) if ts2 is set? No, list.
                                    unique_set.add(s)
                                else:
                                    # It's in both. If count is 1, it's unique to the pair?
                                    # No, if count is 1, it's covered by exactly one block.
                                    # If it's in both, that's a contradiction unless my logic is wrong.
                                    # Wait. subset_counts[s] is the count over ALL selected blocks.
                                    # If s is in b1 and b2, and count is 1, that's impossible.
                                    # So if s is in both, count >= 2.
                                    # So we only care about s where count == 1.
                                    # These s are NOT in any other block.
                                    # So if s is in ts1 and count == 1, it is unique to b1.
                                    # It cannot be in ts2.
                                    pass
                        # So:
                        for s in ts1:
                            if subset_counts[s] == 1:
                                unique_set.add(s)
                        for s in ts2:
                            if subset_counts[s] == 1:
                                unique_set.add(s)
                                
                        if not unique_set:
                            # Both blocks are redundant?
                            # Should have been caught by redundancy removal.
                            # But maybe counts changed?
                            # Remove both?
                            # Let's just remove one and let the loop catch the other.
                            pass
                        else:
                            # Find a block that covers all unique_set
                            # Intersection of t_to_blocks_sets for each s in unique_set
                            if len(unique_set) > 5:
                                # Too many unique subsets, unlikely to find a single block
                                continue
                                
                            candidates = set()
                            first = True
                            for s in unique_set:
                                if first:
                                    candidates = set(t_to_blocks_sets[s])
                                    first = False
                                else:
                                    candidates.intersection_update(t_to_blocks_sets[s])
                                    
                            candidates.difference_update(selected_set)
                            
                            if candidates:
                                # Pick a candidate
                                # Prefer one with low waste
                                cand_list = list(candidates)
                                if len(cand_list) > 10:
                                    cand_list = rng.sample(cand_list, 10)
                                    
                                best_cand = None
                                best_cost = float('inf')
                                for new_bi in cand_list:
                                    new_ts = block_subsets[new_bi]
                                    waste = np.sum(subset_counts[new_ts] > 1)
                                    if waste < best_cost:
                                        best_cost = waste
                                        best_cand = new_bi
                                        
                                if best_cand is not None:
                                    # Perform 2-for-1
                                    # Remove b1, b2
                                    subset_counts[ts1] -= 1
                                    subset_counts[ts2] -= 1
                                    
                                    # Remove from remaining
                                    # Careful with indices
                                    if i1 < i2:
                                        remaining.pop(i2)
                                        remaining.pop(i1)
                                    else:
                                        remaining.pop(i1)
                                        remaining.pop(i2)
                                    
                                    selected_set.remove(b1)
                                    selected_set.remove(b2)
                                    
                                    # Add new_bi
                                    new_ts = block_subsets[best_cand]
                                    subset_counts[new_ts] += 1
                                    remaining.append(best_cand)
                                    selected_set.add(best_cand)
                                    
                                    improved = True
                                    break # Restart local search loop
                            
            # End iterations loop
            if time.time() > deadline - 5.0:
                break

        return remaining

    # Main GRASP Loop
    best_solution = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple GRASP iterations
    # Start with standard greedy (alpha=0)
    # Then try various alpha values
    
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    
    # Shuffle alphas to diversify
    rng.shuffle(alphas)
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
            
        # Construct
        selected = greedy_construct(alpha, rng)
        
        if not selected:
            continue
            
        # Local Search
        selected = local_search(selected, rng)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected.copy()
            
        # Optional: Try to improve further with more local search passes?
        # The local search already loops until no improvement.
        
    # If we haven't found a solution (shouldn't happen), fall back to pure greedy
    if best_solution is None:
        best_solution = greedy_construct(0.0, rng)
        
    # Final verification (optional, but good for safety)
    # Check if best_solution covers all subsets
    # final_covered = 0
    # for idx in best_solution:
    #     final_covered |= block_masks[idx]
    # if final_covered != all_mask:
    #     # Fallback to pure greedy
    #     best_solution = greedy_construct(0.0, rng)
        
    # Convert indices to blocks
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks
