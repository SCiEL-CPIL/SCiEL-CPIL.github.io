import itertools
import sys
import time

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    block_masks = [0] * n_k_blocks
    
    # Precompute blocks containing each t-subset for local search
    # blocks_containing_sub[s_idx] = list of block indices covering s_idx
    blocks_containing_sub = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            blocks_containing_sub[idx].append(i)
        block_masks[i] = mask
        
    # 3. Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    selected_block_indices = []
    selected_set = set()
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected_block_indices.append(best_idx)
        selected_set.add(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # Helper to update counts and unique masks
    def update_counts_and_unique():
        counts = [0] * n_t_subs
        unique_masks = [0] * n_k_blocks # Only valid for selected blocks, but size n_k_blocks for easy access? No, just list for selected.
        
        # We need unique_masks corresponding to selected_block_indices
        # Let's create a dict or parallel list.
        # Since we modify selected_block_indices, let's just compute on the fly in the loop.
        
        # First pass: counts
        for idx in selected_block_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
        
        # Second pass: unique masks for selected blocks
        # unique_mask[i] = mask of bits in block_masks[i] where counts[bit] == 1
        # We store these in a list parallel to selected_block_indices
        unique_masks_selected = []
        for idx in selected_block_indices:
            m = block_masks[idx]
            u_mask = 0
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    u_mask |= lsb
                m ^= lsb
            unique_masks_selected.append(u_mask)
            
        return counts, unique_masks_selected

    # Local Search Loop
    # We will iterate until no improvement or time limit
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for safety
    
    while time.time() - start_time < time_limit:
        counts, unique_masks_selected = update_counts_and_unique()
        improved = False
        
        # 1. Prune redundant blocks (Drop)
        # If unique_mask is 0, the block is redundant
        new_selected = []
        new_selected_set = set()
        for i, idx in enumerate(selected_block_indices):
            if unique_masks_selected[i] != 0:
                new_selected.append(idx)
                new_selected_set.add(idx)
            # else: drop it
        
        if len(new_selected) < len(selected_block_indices):
            selected_block_indices = new_selected
            selected_set = new_selected_set
            improved = True
            continue # Restart loop to recompute counts
        
        # 2. 2-Swap Local Search
        # Try to replace two selected blocks with one new block
        n_sel = len(selected_block_indices)
        found_swap = False
        
        # To speed up, we can iterate pairs. 
        # If n_sel is large, this is O(N^2). 
        # We break on first found swap to keep it simple and fast per iteration.
        for i in range(n_sel):
            if found_swap: break
            idx_i = selected_block_indices[i]
            u_mask_i = unique_masks_selected[i]
            
            for j in range(i + 1, n_sel):
                if found_swap: break
                idx_j = selected_block_indices[j]
                u_mask_j = unique_masks_selected[j]
                
                # Subsets that would be uncovered if we remove i and j
                # These are subsets covered ONLY by i or ONLY by j
                u_mask_union = u_mask_i | u_mask_j
                
                if u_mask_union == 0:
                    # Both are redundant? Should have been pruned.
                    continue
                
                # Find a candidate block b that covers u_mask_union
                # Pick the most constrained subset in u_mask_union
                # To find min degree subset, we check popcount of blocks_containing_sub[s]
                # We iterate through bits in u_mask_union
                
                m = u_mask_union
                best_candidate_block = -1
                min_degree = float('inf')
                best_sub_idx = -1
                
                while m:
                    lsb = m & -m
                    s_idx = lsb.bit_length() - 1
                    deg = len(blocks_containing_sub[s_idx])
                    if deg < min_degree:
                        min_degree = deg
                        best_sub_idx = s_idx
                        # Optimization: if deg is 1, that's the best we can get
                        if deg == 1:
                            break
                    m ^= lsb
                
                if best_sub_idx == -1:
                    continue
                
                # Check candidates containing best_sub_idx
                for b in blocks_containing_sub[best_sub_idx]:
                    if b in selected_set:
                        continue
                    
                    # Check if b covers all of u_mask_union
                    if (u_mask_union & ~block_masks[b]) == 0:
                        # Found a valid replacement!
                        # Remove i and j, add b
                        selected_block_indices.pop(j) # Remove j first (higher index)
                        selected_block_indices.pop(i) # Remove i
                        selected_set.remove(idx_i)
                        selected_set.remove(idx_j)
                        
                        selected_block_indices.append(b)
                        selected_set.add(b)
                        
                        found_swap = True
                        improved = True
                        break # Break out of b loop
                
                if found_swap:
                    break # Break out of j loop
            
            if found_swap:
                break # Break out of i loop
        
        if found_swap:
            continue # Restart loop to recompute counts
        
        # If no prune and no swap, we are stuck in local optimum
        # We could try 1-swap or random restarts, but for now, stop.
        # A simple 1-swap (replace 1 with 1) doesn't reduce count, so it's not useful for minimization unless it enables a future 2-swap.
        # Given time constraints, we stop here.
        break

    # Final Prune just in case
    counts, unique_masks_selected = update_counts_and_unique()
    final_indices = []
    for i, idx in enumerate(selected_block_indices):
        if unique_masks_selected[i] != 0:
            final_indices.append(idx)
    
    return [list(k_blocks[i]) for i in final_indices]
