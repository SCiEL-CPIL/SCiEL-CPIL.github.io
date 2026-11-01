import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each block, store the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # For each t-subset, store the list of blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # 2. Greedy Set Cover
    uncovered = [True] * num_t_subsets
    gain = [0] * num_k_subsets
    
    # Initialize gain
    for b_idx in range(num_k_subsets):
        gain[b_idx] = len(block_t_indices[b_idx])
        
    selected_blocks = []
    
    while True:
        if time.time() > deadline:
            break
        max_gain = max(gain)
        if max_gain == 0:
            break
            
        best_block = gain.index(max_gain)
        
        selected_blocks.append(best_block)
        
        # Update gains
        for t_idx in block_t_indices[best_block]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= 1
                    
    # 3. Minimalization (Local Search: Deletion)
    # Calculate cover counts
    cover_count = [0] * num_t_subsets
    for b_idx in selected_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    active_blocks = set(selected_blocks)
    
    # Multiple passes of deletion to find more removable blocks
    # We do up to 3 passes, or until no more blocks can be removed
    for _pass in range(3):
        if time.time() > deadline:
            break
        indices_to_check = list(active_blocks)
        random.shuffle(indices_to_check)
        removed_any = False
        
        for b_idx in indices_to_check:
            if b_idx not in active_blocks:
                continue
                
            can_remove = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
                    
            if can_remove:
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                removed_any = True
                
        if not removed_any:
            break
            
    # 4. Local Search: Swap
    # Try to swap one block for another to reduce total count or improve "tightness"
    # We look for blocks that can be removed if we add a different block that covers the unique subsets.
    # This is complex. A simpler heuristic:
    # Try to replace a block with another that covers more "critical" subsets.
    # Given time constraints, we'll stick to deletion for now but add a simple swap heuristic.
    
    # For each active block, check if there's a non-active block that covers all its unique subsets (cover_count==1)
    # and has higher "value" (covers more uncovered or critical subsets).
    # Actually, since we are minimizing, we only swap if it allows removal of another block later?
    # Let's try a simpler improvement:
    # Randomly shuffle and try to swap a block with a better one if it doesn't increase size.
    # But we can't increase size. So we only swap if it allows subsequent deletion.
    
    # Let's try a targeted swap:
    # Identify blocks with "unique" subsets (cover_count == 1).
    # If we can find a new block that covers all unique subsets of an existing block,
    # we might be able to remove the existing block.
    
    # This is O(N^2) potentially. Let's do a limited search.
    
    # First, identify blocks with unique subsets
    blocks_with_unique = []
    for b_idx in active_blocks:
        has_unique = False
        for t_idx in block_t_indices[b_idx]:
            if cover_count[t_idx] == 1:
                has_unique = True
                break
        if has_unique:
            blocks_with_unique.append(b_idx)
    
    # For each block with unique subsets, try to find a replacement
    # We sample a few random non-active blocks to see if they cover the unique subsets
    # This is a heuristic improvement
    
    random.shuffle(blocks_with_unique)
    
    for b_idx in blocks_with_unique:
        if time.time() > deadline:
            break
            
        # Get the unique subsets for this block
        unique_subsets = [t_idx for t_idx in block_t_indices[b_idx] if cover_count[t_idx] == 1]
        
        if not unique_subsets:
            continue
            
        # Try to find a block that covers all these unique subsets
        # We sample 100 random non-active blocks
        # If found, we add it, remove the current one, and check if cover counts are valid
        # But adding a block increases count by 1, removing decreases by 1. Net change 0.
        # We need to ensure the new block doesn't break anything? No, it only adds coverage.
        # The risk is that it might not allow further reductions.
        # But if it covers the unique subsets, the old block becomes redundant (cover_count > 1 for all its subsets).
        # So we can remove the old block.
        
        # We need to be careful: if the new block is already in active_blocks, skip.
        # We need to pick a block NOT in active_blocks.
        
        # This is expensive to check for every unique block.
        # Let's just try a few swaps.
        
        # Sample 50 random blocks from all_k_subsets
        # Check if any is not in active_blocks and covers all unique_subsets
        
        found_replacement = False
        for _ in range(50):
            new_b_idx = random.randint(0, num_k_subsets - 1)
            if new_b_idx in active_blocks:
                continue
                
            # Check if new_b_idx covers all unique_subsets
            # We need to check if each t_idx in unique_subsets is in block_t_indices[new_b_idx]
            # This is O(len(unique_subsets) * log(len(block_t_indices[new_b_idx]))) if we use sets
            # But block_t_indices are lists.
            # Let's convert to set for checking? Or just iterate.
            
            # Optimization: Precompute sets for block_t_indices?
            # That would take memory.
            # Let's just check by iterating unique_subsets and seeing if they are in the new block's list.
            # To speed up, we can check the first unique subset.
            
            # Actually, a better way:
            # For each t_idx in unique_subsets, check if new_b_idx is in t_subset_blocks[t_idx]
            
            valid = True
            for t_idx in unique_subsets:
                if new_b_idx not in t_subset_blocks[t_idx]:
                    valid = False
                    break
                    
            if valid:
                # We found a replacement
                # Add new_b_idx, remove b_idx
                active_blocks.add(new_b_idx)
                active_blocks.remove(b_idx)
                
                # Update cover counts
                # Decrement for b_idx
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                # Increment for new_b_idx
                for t_idx in block_t_indices[new_b_idx]:
                    cover_count[t_idx] += 1
                    
                found_replacement = True
                break
                
        if found_replacement:
            # After a swap, some blocks might become removable.
            # We don't run full deletion again here to save time, but the next outer pass (if any) will.
            # Or we can do a quick check on the newly added block? No, it's new.
            # We can do a quick check on the remaining active blocks.
            # Let's just break out and let the next deletion pass handle it.
            # But we are in a loop over blocks_with_unique.
            # Let's just continue to the next unique block.
            continue

    # Final deletion pass after swaps
    if time.time() < deadline:
        indices_to_check = list(active_blocks)
        random.shuffle(indices_to_check)
        
        for b_idx in indices_to_check:
            if b_idx not in active_blocks:
                continue
                
            can_remove = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
                    
            if can_remove:
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1

    # Convert back to list of lists
    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    
    return result
