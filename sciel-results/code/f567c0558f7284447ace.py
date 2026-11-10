import itertools
import time
import random

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy Set Cover followed by
    iterative Greedy Deletion.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage: for each block, which t-subsets does it cover?
    # block_covers[b] is a list of indices of t-subsets covered by block b
    block_covers = []
    for block in all_blocks:
        covered_indices = []
        for t_sub in itertools.combinations(block, t):
            idx = subset_to_idx[t_sub]
            covered_indices.append(idx)
        block_covers.append(covered_indices)
    
    # Greedy Construction
    # coverage_count[i] = number of blocks in current solution that cover t-subset i
    coverage_count = [0] * n_t_subsets
    current_blocks = []
    
    # For speed, we can maintain a list of uncovered subsets
    # But calculating gain for all blocks every step is O(n_blocks * avg_cover_size)
    # n_blocks for C(19,7) is 50388. n_t_subsets for C(19,4) is 3876.
    # This is feasible.
    
    while True:
        if time.time() > deadline:
            break
        
        # Find block with max gain
        max_gain = 0
        best_block_idx = -1
        
        # Optimization: Only consider blocks that cover at least one uncovered subset
        # We can just iterate all blocks and sum coverage_count[i] == 0 for i in block_covers[b]
        
        for b_idx in range(n_blocks):
            gain = 0
            for t_idx in block_covers[b_idx]:
                if coverage_count[t_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
        
        if max_gain == 0:
            # All subsets covered
            break
        
        # Add best block
        current_blocks.append(best_block_idx)
        for t_idx in block_covers[best_block_idx]:
            coverage_count[t_idx] += 1
        
        if time.time() > deadline:
            break

    # Convert indices to actual blocks
    solution_blocks = [all_blocks[i] for i in current_blocks]
    
    # Local Search: Greedy Deletion
    # Try to remove blocks one by one if coverage is maintained
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Make a copy to iterate
        blocks_to_try = list(solution_blocks)
        
        for i in range(len(blocks_to_try)):
            if time.time() > deadline:
                break
            
            block = blocks_to_try[i]
            # Tentatively remove block
            # We need to check if all t-subsets are still covered by the remaining blocks
            
            # To do this efficiently, we can recompute coverage for the remaining blocks
            # Or, we can count how many times each t-subset is covered by the current set.
            
            # Let's recompute coverage counts for the set excluding this block
            # This is O(N_blocks * avg_cover_size) which is okay if N_blocks is ~150
            
            # Actually, simpler: 
            # 1. Count coverage of all t-subsets by ALL current blocks.
            # 2. For the block being removed, subtract its contribution.
            # 3. If any count becomes 0, removal fails.
            
            # Let's maintain a global coverage count array for the current solution
            # This is tricky because we are modifying the list.
            
            # Alternative: Just recompute from scratch for each candidate removal?
            # N_blocks ~ 150. N_t_subsets ~ 3876.
            # For each removal candidate:
            #   - Build set of t-subsets covered by remaining blocks.
            #   - Check if it equals all_t_subsets.
            
            # This is O(N_blocks * N_t_subsets) per check? No.
            # Let's use a boolean array `covered_by_others`.
            
            # Optimization: Precompute for each block the set of t-subsets it covers.
            # block_t_sets = {block: set of t-subsets}
            
            # Let's just do a direct check for simplicity and correctness first, then optimize if needed.
            
            # Create a set of all t-subsets covered by (current_blocks - {block})
            # We can do this by iterating all other blocks and adding their t-subsets to a set.
            
            remaining_covered = set()
            for j, other_block in enumerate(solution_blocks):
                if j == i:
                    continue
                for t_sub in itertools.combinations(other_block, t):
                    remaining_covered.add(t_sub)
            
            if len(remaining_covered) == n_t_subsets:
                # Safe to remove
                solution_blocks.pop(i)
                improved = True
                # Restart the loop from beginning as indices changed
                break
            
            # If we didn't break, we kept the block

    return solution_blocks

# The above Greedy Deletion is O(N^2 * C(k,t)) which might be slow if N is large.
# Let's refine the deletion to be faster.
# We can maintain a count of how many blocks cover each t-subset.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage indices
    block_covers = []
    for block in all_blocks:
        covered_indices = [subset_to_idx[t_sub] for t_sub in itertools.combinations(block, t)]
        block_covers.append(covered_indices)
    
    # Greedy Construction
    coverage_count = [0] * n_t_subsets
    current_block_indices = []
    
    while True:
        if time.time() > deadline:
            break
        
        max_gain = 0
        best_block_idx = -1
        
        for b_idx in range(n_blocks):
            gain = 0
            for t_idx in block_covers[b_idx]:
                if coverage_count[t_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
        
        if max_gain == 0:
            break
        
        current_block_indices.append(best_block_idx)
        for t_idx in block_covers[best_block_idx]:
            coverage_count[t_idx] += 1
        
        if time.time() > deadline:
            break

    # Local Search: Greedy Deletion with maintained counts
    # We have coverage_count reflecting current_block_indices.
    # We want to try removing blocks.
    
    # To speed up deletion, we can iterate and try to remove.
    # If we remove block B, we decrement counts for its t-subsets.
    # If any count hits 0, we fail and increment back.
    
    solution_indices = list(current_block_indices)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for i in range(len(solution_indices)):
            if time.time() > deadline:
                break
            
            b_idx = solution_indices[i]
            
            # Tentatively remove
            # Check if any t-subset count becomes 0
            valid = True
            for t_idx in block_covers[b_idx]:
                coverage_count[t_idx] -= 1
                if coverage_count[t_idx] == 0:
                    valid = False
                    break
            
            if valid:
                # Successfully removed
                solution_indices.pop(i)
                improved = True
                # Restart loop
                break
            else:
                # Restore counts
                for t_idx in block_covers[b_idx]:
                    coverage_count[t_idx] += 1
                # Note: We broke out of the t_idx loop early if invalid, 
                # but we need to ensure all decremented ones are restored.
                # The logic above restores only if we didn't break? 
                # No, if valid is False, we break out of the inner loop.
                # We need to restore all decrements.
                # Let's rewrite the check to be safe.
                
    # The restoration logic above is flawed if we break early.
    # Let's implement a safer version.
    
    # Re-implement deletion safely
    solution_indices = list(current_block_indices)
    # Rebuild coverage counts for this list
    coverage_count = [0] * n_t_subsets
    for b_idx in solution_indices:
        for t_idx in block_covers[b_idx]:
            coverage_count[t_idx] += 1
            
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for i in range(len(solution_indices)):
            if time.time() > deadline:
                break
            
            b_idx = solution_indices[i]
            
            # Check if removal is valid
            # A removal is valid if no t-subset has count 1 and is covered ONLY by this block.
            # i.e., for all t_idx in block_covers[b_idx], coverage_count[t_idx] > 1
            
            can_remove = True
            for t_idx in block_covers[b_idx]:
                if coverage_count[t_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Perform removal
                for t_idx in block_covers[b_idx]:
                    coverage_count[t_idx] -= 1
                solution_indices.pop(i)
                improved = True
                break

    return [all_blocks[i] for i in solution_indices]
