import itertools
import time
import random
import sys

# Precomputation for v=19, k=8, t=5
# These are constants for the specific task, but we should probably compute them 
# or hardcode the logic to be generic if possible, though the task specifies 
# "evaluator passes this task's parameters v, k and t". 
# However, the precomputation of all blocks and subsets is specific to v, k, t.
# Given the small size, we can compute inside construct or globally if v,k,t are fixed.
# The prompt says "evaluator passes this task's parameters v, k and t".
# So we must handle general v, k, t, but the performance relies on small v.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # 1. Generate all t-subsets and assign them indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {subset: i for i, subset in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # 3. Precompute bitmask for each k-block
    # A bit i is set if the i-th t-subset is contained in this k-block
    block_masks = [0] * num_k_blocks
    for b_idx, block in enumerate(k_blocks):
        mask = 0
        # Check all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        
    # 4. Precompute mapping from t-subset to list of block indices containing it
    subset_to_blocks = {subset: [] for subset in t_subsets}
    for b_idx, block in enumerate(k_blocks):
        for t_sub in itertools.combinations(block, t):
            subset_to_blocks[t_sub].append(b_idx)
            
    # 5. Greedy Algorithm with Random Restarts
    best_solution = None
    best_size = float('inf')
    
    # Initial full mask
    full_mask = (1 << num_t_subsets) - 1
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = full_mask
        selected_blocks = []
        
        # To speed up selection, we can maintain a list of uncovered subset indices
        # But picking a random uncovered subset from the bitmask is tricky.
        # We can keep a set of uncovered indices, but that's slow to update.
        # Alternative: Pick a random block? No, we need to pick a subset to drive the search.
        
        # Strategy: Pick a random uncovered t-subset.
        # To do this efficiently, we can maintain a list of uncovered indices and 
        # occasionally rebuild it, or just pick a random index and check if it's uncovered.
        
        # Let's use a list of uncovered indices for efficient random selection.
        uncovered_indices = list(range(num_t_subsets))
        
        while uncovered_indices:
            # Pick a random uncovered t-subset
            idx = random.randint(0, len(uncovered_indices) - 1)
            subset_idx = uncovered_indices[idx]
            subset = t_subsets[subset_idx]
            
            # Get candidate blocks containing this subset
            candidates = subset_to_blocks[subset]
            
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in candidates:
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            # Select the best block
            selected_blocks.append(best_block_idx)
            
            # Update uncovered mask
            uncovered &= ~block_masks[best_block_idx]
            
            # Update uncovered_indices
            # This is the expensive part. We need to remove all indices that are now covered.
            # We can do this by filtering the list.
            # Since num_t_subsets is ~11k, and we do this ~300 times, 
            # filtering 11k items 300 times is 3.3M ops, which is fine.
            # But we can be smarter: only remove those that were in the block.
            
            # Let's just rebuild the list from the mask? 
            # No, iterating 11k bits to build a list is O(11k).
            # Filtering the existing list is also O(N).
            
            # Optimization: Instead of maintaining a list, we can just pick a random 
            # index from 0..num_t_subsets-1 and check if it's uncovered.
            # If not, try again.
            # As coverage increases, the probability of picking an uncovered one drops.
            # But initially it's 1.0, later it's small.
            # This "Rejection Sampling" might be slow if many are covered.
            
            # Let's stick to the list update, but optimize it.
            # We know which bits were cleared.
            # We can iterate through the t-subsets of the selected block and 
            # remove their indices from uncovered_indices.
            
            # Map t-subset to index is already done.
            # We need to remove specific indices from a list.
            # Using a set for uncovered_indices would make removal O(1), but iteration for random pick is O(N) if we pick random from set? No, random from set is not direct.
            # We can maintain a list and a boolean array "is_uncovered".
            
            # Let's restructure:
            # is_uncovered = [True] * num_t_subsets
            # uncovered_indices = list(range(num_t_subsets))
            
            # When we select a block:
            # for t_sub in itertools.combinations(block, t):
            #     idx = t_subset_index[t_sub]
            #     if is_uncovered[idx]:
            #         is_uncovered[idx] = False
            #         # Remove idx from uncovered_indices efficiently?
            #         # Swapping with last element is O(1) if we know the position.
            
            # Let's use the swap-remove method.
            # pos_map[idx] = position in uncovered_indices
            
            pass # Placeholder for logic refinement in actual code below

    # Refactored Greedy with Swap-Remove
    def greedy_cover_optimized(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        is_uncovered = [True] * num_t_subsets
        uncovered_indices = list(range(num_t_subsets))
        pos_map = {i: i for i in range(num_t_subsets)}
        
        selected_blocks = []
        
        while uncovered_indices:
            # Pick random index from uncovered_indices
            rand_pos = random.randint(0, len(uncovered_indices) - 1)
            subset_idx = uncovered_indices[rand_pos]
            subset = t_subsets[subset_idx]
            
            # Find best block covering this subset
            candidates = subset_to_blocks[subset]
            
            best_gain = -1
            best_block_idx = -1
            
            # We can pre-calculate the mask of uncovered bits? No, we use the global mask.
            # But we don't have a global mask in this version, we have is_uncovered.
            # To calculate gain, we need to count how many t-subsets in the block are uncovered.
            # We can iterate through the t-subsets of the candidate block and check is_uncovered.
            
            for b_idx in candidates:
                block = k_blocks[b_idx]
                gain = 0
                for t_sub in itertools.combinations(block, t):
                    idx = t_subset_index[t_sub]
                    if is_uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            selected_blocks.append(best_block_idx)
            
            # Update is_uncovered and uncovered_indices
            block = k_blocks[best_block_idx]
            for t_sub in itertools.combinations(block, t):
                idx = t_subset_index[t_sub]
                if is_uncovered[idx]:
                    is_uncovered[idx] = False
                    # Remove idx from uncovered_indices using pos_map
                    pos = pos_map[idx]
                    last_idx = uncovered_indices[-1]
                    uncovered_indices[pos] = last_idx
                    pos_map[last_idx] = pos
                    uncovered_indices.pop()
                    del pos_map[idx]
                    
        return selected_blocks

    # Local Search: Remove redundant blocks
    def local_search(blocks):
        if not blocks:
            return blocks
        
        # Count coverage
        coverage_counts = [0] * num_t_subsets
        for b_idx in blocks:
            block = k_blocks[b_idx]
            for t_sub in itertools.combinations(block, t):
                idx = t_subset_index[t_sub]
                coverage_counts[idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                block = k_blocks[b_idx]
                redundant = True
                for t_sub in itertools.combinations(block, t):
                    idx = t_subset_index[t_sub]
                    if coverage_counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for t_sub in itertools.combinations(block, t):
                        idx = t_subset_index[t_sub]
                        coverage_counts[idx] -= 1
                    blocks.pop(i)
                    changed = True
        
        return blocks

    # Run multiple restarts
    attempts = 0
    while time.time() < deadline:
        seed = random.randint(0, 2**32 - 1)
        solution = greedy_cover_optimized(seed)
        solution = local_search(solution)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution
            # print(f"New best: {best_size}", file=sys.stderr) # Debug
            
        attempts += 1
        # If we've run enough attempts, we might break early if we hit a lower bound?
        # Lower bound is 264. If we hit 335, we can't do better than known best easily.
        # But we can try to go lower.

    if best_solution is None:
        # Fallback to simple deterministic greedy if something went wrong or no time
        return []
        
    # Convert indices back to lists of points
    final_blocks = [list(k_blocks[b_idx]) for b_idx in best_solution]
    return final_blocks
