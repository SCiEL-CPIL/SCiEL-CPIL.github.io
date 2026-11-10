import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Enumerate all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to its index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute bitsets for all k-subsets (blocks)
    # There are C(v, k) blocks
    all_k_subsets = list(itertools.combinations(range(v), k))
    block_bitsets = []
    
    for block in all_k_subsets:
        bits = 0
        # For each t-subset within this block, set the corresponding bit
        for t_sub in itertools.combinations(block, t):
            idx = subset_to_idx[t_sub]
            bits |= (1 << idx)
        block_bitsets.append(bits)
    
    # 3. Greedy Set Cover
    # uncovered is a bitmask of all t-subsets initially
    # Since num_t_subsets can be > 63, we use Python's arbitrary precision int
    uncovered = (1 << num_t_subsets) - 1
    selected_blocks = []
    used_block_indices = set()
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_idx = -1
        
        # Evaluate all candidate blocks
        # Optimization: We can skip blocks that are subsets of already selected blocks? No, they are all size k.
        # We can skip blocks that have 0 gain.
        
        for i, bits in enumerate(block_bitsets):
            # Calculate gain: number of new t-subsets covered
            gain = (bits & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Early termination if we found a block that covers everything? 
                # Unlikely in one step, but if gain == uncovered.bit_count(), we are done.
                if gain == uncovered.bit_count():
                    break
        
        if best_gain <= 0:
            # Should not happen if uncovered is not 0 and we have valid blocks
            break
            
        selected_blocks.append(all_k_subsets[best_idx])
        used_block_indices.add(best_idx)
        
        # Update uncovered
        uncovered &= ~block_bitsets[best_idx]
        
        # Optional: Check if we should stop early if time is tight, but we are likely fast.
        
    # 4. Redundancy Removal
    # Iterate through selected blocks and remove any that are redundant.
    # A block is redundant if all its t-subsets are covered by the union of other blocks.
    # This is O(N^2 * T) where N is num blocks, T is t-subsets per block.
    # N ~ 150, T ~ 35. 150^2 * 35 ~ 800k operations. Very fast.
    
    final_blocks = list(selected_blocks)
    
    # We need to be careful: removing a block might make another one redundant? 
    # Usually, we do a single pass or iterate until stable.
    # Let's do a simple pass: 
    # For each block, check if it's covered by the rest.
    
    # To check if block B is redundant, we need to know if every t-subset in B is in some other block.
    # We can precompute the "coverage count" for each t-subset from the final set.
    
    # Let's try a simpler heuristic: 
    # Sort blocks by "uniqueness" (number of t-subsets they cover that no other block covers).
    # But calculating uniqueness is expensive.
    
    # Alternative: Just do a greedy removal pass.
    # While True:
    #   found_redundant = False
    #   for i in range(len(final_blocks)):
    #      if is_redundant(final_blocks[i], final_blocks[:i] + final_blocks[i+1:]):
    #          final_blocks.pop(i)
    #          found_redundant = True
    #          break
    #   if not found_redundant: break
    
    # is_redundant check:
    #   union_of_others = OR of bitsets of other blocks
    #   return (bitset_of_block & union_of_others) == bitset_of_block
    
    # This is O(N^2) big-int ops. 150^2 = 22500 ops. Fast.
    
    # Let's implement this refinement.
    
    # We need the bitsets for the final blocks.
    # We already have block_bitsets for all possible blocks.
    # Let's map final_blocks to their bitsets.
    
    # Actually, we can just use the indices.
    # selected_block_indices is the list of indices into all_k_subsets.
    
    # Let's convert selected_blocks back to indices?
    # No, we have `used_block_indices` but it's a set, order is lost.
    # Let's keep a list of indices.
    
    # Re-do greedy to keep indices? Or just re-lookup.
    # Re-lookup is O(N * log M). N=150, M=50000. Fast.
    
    block_to_idx = {b: i for i, b in enumerate(all_k_subsets)}
    selected_indices = [block_to_idx[b] for b in final_blocks]
    
    # Refinement loop
    changed = True
    while changed:
        changed = False
        for i in range(len(selected_indices)):
            if time.time() > deadline:
                break
            
            # Check if selected_indices[i] is redundant given the others
            idx_i = selected_indices[i]
            bits_i = block_bitsets[idx_i]
            
            # Compute union of others
            union_others = 0
            for j in range(len(selected_indices)):
                if j != i:
                    union_others |= block_bitsets[selected_indices[j]]
            
            if (bits_i & ~union_others) == 0:
                # Redundant
                selected_indices.pop(i)
                changed = True
                break # Restart loop from 0 since indices shifted
        
        if time.time() > deadline:
            break
            
    # Convert indices back to blocks
    result_blocks = [all_k_subsets[i] for i in selected_indices]
    
    return result_blocks
