import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precomputation
    # Generate all t-subsets and assign each a compact index 0..N_t-1
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Precompute for each k-block:
    #   block_cover_mask: bitmask over t-subset indices
    #   block_t_indices: list of t-subset indices covered
    block_cover_mask = [0] * n_k
    block_t_indices = [[] for _ in range(n_k)]
    
    for bi, block in enumerate(all_k_subsets):
        cm = 0
        tlist = []
        for t_sub in itertools.combinations(block, t):
            ti = t_sub_to_idx[t_sub]
            cm |= (1 << ti)
            tlist.append(ti)
        block_cover_mask[bi] = cm
        block_t_indices[bi] = tlist

    # For each t-subset index, list of block indices that cover it
    # This is useful for finding candidates for swaps
    t_idx_to_blocks = [[] for _ in range(n_t)]
    for bi in range(n_k):
        for ti in block_t_indices[bi]:
            t_idx_to_blocks[ti].append(bi)

    # 2. Search
    best_blocks = None
    best_count = float('inf')
    full_mask = (1 << n_t) - 1

    def greedy_once(rng):
        """Run one randomized greedy pass."""
        uncovered_mask = full_mask
        blocks = []
        
        while uncovered_mask:
            # Find block with max gain
            max_gain = 0
            best_candidates = []
            
            # Iterate over all blocks
            # Optimization: If we had a priority queue or similar, we could be faster,
            # but n_k is ~50k, which is small.
            for bi in range(n_k):
                gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [bi]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(bi)
            
            if not best_candidates:
                return None # Should not happen if covering exists
            
            # Randomize choice among best candidates to explore different paths
            chosen_bi = rng.choice(best_candidates)
            blocks.append(chosen_bi)
            uncovered_mask &= ~block_cover_mask[chosen_bi]
            
        return blocks

    def local_search(block_indices, time_limit):
        """
        Refine the set of blocks by removing redundant ones and performing swaps.
        Returns the refined list of block indices.
        """
        if block_indices is None:
            return None
            
        block_set = set(block_indices)
        current_count = len(block_set)
        
        # Build coverage counts
        cover_count = [0] * n_t
        for bi in block_set:
            for ti in block_t_indices[bi]:
                cover_count[ti] += 1

        improved = True
        iterations = 0
        max_iterations = 100 # Safety cap for local search iterations
        
        while improved and time.time() < time_limit and iterations < max_iterations:
            improved = False
            iterations += 1

            # 1. Try removing redundant blocks
            # We iterate through a copy of the set to allow removal
            for bi in list(block_set):
                # Check if removing bi leaves any t-subset uncovered
                removable = True
                for ti in block_t_indices[bi]:
                    if cover_count[ti] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove it
                    block_set.remove(bi)
                    for ti in block_t_indices[bi]:
                        cover_count[ti] -= 1
                    current_count -= 1
                    improved = True
                    # We can break out of the removal loop to restart the check,
                    # as removals change the cover counts significantly
                    break 

            # 2. Try swaps if no removal was possible
            if not improved and time.time() < time_limit:
                # Try to find a swap that might enable future removals
                # Strategy: For each block, find a replacement that covers all unique
                # t-subsets of the current block.
                # To improve quality, we can prefer replacements that cover more
                # "slack" (t-subsets with count > 1).
                
                swaps_made = False
                # Shuffle blocks to randomize swap order
                blocks_list = list(block_set)
                random.shuffle(blocks_list)
                
                for bi in blocks_list:
                    if time.time() > time_limit:
                        break
                        
                    # Find t-subsets uniquely covered by bi
                    unique_to_bi = []
                    for ti in block_t_indices[bi]:
                        if cover_count[ti] == 1:
                            unique_to_bi.append(ti)
                    
                    if not unique_to_bi:
                        # This block is completely redundant? 
                        # If so, it should have been caught by removal.
                        # But maybe cover_count logic needs refresh? 
                        # Actually, if unique_to_bi is empty, all its t-subsets are covered by others.
                        # So it is removable.
                        block_set.remove(bi)
                        for ti in block_t_indices[bi]:
                            cover_count[ti] -= 1
                        current_count -= 1
                        improved = True
                        swaps_made = True
                        break

                    # Find candidate blocks that cover all unique_to_bi
                    # Candidate must be in the intersection of t_idx_to_blocks for all ti in unique_to_bi
                    # Since unique_to_bi is small (max 35), we can intersect lists
                    candidate_set = set(t_idx_to_blocks[unique_to_bi[0]])
                    for ti in unique_to_bi[1:]:
                        candidate_set &= set(t_idx_to_blocks[ti])
                    
                    # Exclude current block
                    candidate_set.discard(bi)
                    
                    if not candidate_set:
                        continue
                        
                    # Evaluate candidates
                    best_candidate = None
                    best_score = -1
                    
                    for cb in candidate_set:
                        # Calculate score: how many t-subsets does cb cover that are "slack" (count > 1)?
                        # Or simply: does it cover more total?
                        # Let's use a heuristic: prefer blocks that cover more t-subsets with count >= 2
                        # This increases robustness.
                        
                        # Quick check: if cb is already in block_set, skip (swapping with self is useless)
                        if cb in block_set:
                            continue
                            
                        # Score: number of t-subsets in cb that are currently covered by >= 2 blocks
                        # This is a proxy for "slack"
                        slack_gain = 0
                        for ti in block_t_indices[cb]:
                            if cover_count[ti] >= 2:
                                slack_gain += 1
                            
                        # Also penalize if cb doesn't cover some t-subsets that bi covers 
                        # (but we already ensured it covers unique ones)
                        
                        if slack_gain > best_score:
                            best_score = slack_gain
                            best_candidate = cb

                    if best_candidate is not None:
                        # Perform swap
                        block_set.remove(bi)
                        for ti in block_t_indices[bi]:
                            cover_count[ti] -= 1
                        block_set.add(best_candidate)
                        for ti in block_t_indices[best_candidate]:
                            cover_count[ti] += 1
                        improved = True
                        swaps_made = True
                        break # Restart local search after a swap

        return list(block_set)

    # Main Loop
    while time.time() < deadline:
        # Run greedy
        block_indices = greedy_once(random)
        if block_indices is None:
            continue

        # Local search refinement
        # Allocate remaining time for local search, but cap it to allow for restarts
        remaining_time = deadline - time.time()
        ls_time_limit = time.time() + min(remaining_time, 5.0) # Cap local search to 5s per restart
        
        refined = local_search(block_indices, ls_time_limit)
        
        if refined is not None:
            if len(refined) < best_count:
                best_count = len(refined)
                best_blocks = [list(all_k_subsets[bi]) for bi in refined]
                
        # If we have a very good solution, we can try to polish it further
        # but the main loop restarts handle diversity.

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_mask = full_mask
        blocks = []
        while uncovered_mask:
            max_gain = 0
            best_bi = 0
            for bi in range(n_k):
                gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_bi = bi
            blocks.append(all_k_subsets[best_bi])
            uncovered_mask &= ~block_cover_mask[best_bi]
        return [list(b) for b in blocks]

    return best_blocks
