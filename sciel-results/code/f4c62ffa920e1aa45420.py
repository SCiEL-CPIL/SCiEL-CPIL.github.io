import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Use 145s deadline to stay safely within 150s kill limit
    deadline = start_time + 145.0

    # 1. Precomputation
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices for O(1) lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # 2. Build coverage matrix using NumPy for vectorized operations
    # Cover[i, j] is 1 if t-subset j is contained in block i
    # Using uint8 to save memory and allow fast arithmetic
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        # Generate t-subsets from block B
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # 3. Greedy Set Cover Construction
    # We use a vectorized approach to compute gains: gains = Cover @ uncovered
    # This is much faster than iterating in Python.
    
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []
    
    # Leave 10s for post-processing
    greedy_deadline = deadline - 10.0
    
    while np.sum(uncovered) > 0:
        if time.time() > greedy_deadline:
            break
            
        # Compute gain for each block: number of currently uncovered t-subsets it covers
        # Cover is (n_blocks, n_tsubs), uncovered is (n_tsubs,)
        # Result is (n_blocks,)
        gains = Cover @ uncovered
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # No block can cover any uncovered subset (should not happen if valid)
            break
            
        selected_indices.append(best_idx)
        
        # Update uncovered: set to 0 where the best block covers
        # uncovered = uncovered & (Cover[best_idx] == 0)
        # Since Cover[best_idx] is 0/1, ~Cover[best_idx] would be 255/254 in uint8, so use logical AND with boolean mask
        uncovered = uncovered & (Cover[best_idx] == 0)
        
    # 4. Post-processing: Local Search for Minimization
    # We now have a valid covering (assuming greedy finished).
    # We want to minimize the number of blocks.
    
    if time.time() > deadline - 2.0:
        # Not enough time for refinement, return what we have
        return [list(all_kblocks[i]) for i in selected_indices]
    
    # Convert selected_indices to a set for O(1) lookup
    active_blocks = set(selected_indices)
    
    # Compute initial cover counts
    # cover_counts[j] = number of active blocks covering t-subset j
    # We can sum the rows of Cover corresponding to active blocks
    # But summing many rows is slow. Let's do it incrementally or use a sparse-like approach?
    # Actually, n_tsubs is C(19,4) = 3876. n_blocks is C(19,7) = 50388.
    # Summing ~150 rows of length 3876 is 150 * 3876 ~ 580k ops. Very fast.
    
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in active_blocks:
        cover_counts += Cover[idx]
    
    # Helper function to check if a block is removable
    # A block is removable if it doesn't cover any t-subset with count 1.
    # i.e., if min(cover_counts[Cover[idx]==1]) > 1
    
    def is_removable(idx):
        # Get positions where block idx covers
        # We need to check if any of those have count 1
        # Vectorized: np.any((Cover[idx] == 1) & (cover_counts == 1))
        return not np.any((Cover[idx] == 1) & (cover_counts == 1))
    
    # Phase 4a: Deletion Pass
    # Try to remove redundant blocks
    # We do multiple passes because removing one block can make another removable.
    
    deletion_deadline = deadline - 5.0
    pass_count = 0
    while time.time() < deletion_deadline:
        pass_count += 1
        changed = False
        # Shuffle to avoid bias in removal order
        indices_list = list(active_blocks)
        random.shuffle(indices_list)
        
        for idx in indices_list:
            if idx not in active_blocks:
                continue
                
            if is_removable(idx):
                active_blocks.remove(idx)
                cover_counts -= Cover[idx]
                changed = True
                
        if not changed:
            break
            
    # Phase 4b: Swap/Move Search (Local Search)
    # Try to swap a block with another that allows for further reductions.
    # Strategy:
    # 1. Identify "critical" blocks (those with unique coverage).
    # 2. For each critical block, try to find a replacement block that covers its unique subsets.
    # 3. If found, swap, then try to delete more blocks.
    
    swap_deadline = deadline - 2.0
    
    # To make this efficient, we precompute for each t-subset, which blocks cover it.
    # But we already have Cover.
    # t_subset_blocks[t_idx] = list of block indices covering t_idx
    
    # Building this explicitly might be memory heavy (3876 lists of ~150 ints each is fine).
    # Let's build it once.
    t_subset_blocks = [[] for _ in range(n_tsubs)]
    for j in range(n_tsubs):
        # Find all i such that Cover[i, j] == 1
        # This is a column search.
        # Since we do this once, it's okay.
        # But doing it in a loop in Python is slow.
        # Alternative: use np.where on columns?
        # Or just iterate over active_blocks? No, we need ALL blocks for replacement.
        
        # Let's just iterate over all blocks for each t-subset?
        # 3876 * 50388 checks is ~195M checks. Too slow in Python.
        
        # Better: Invert the matrix construction logic or use scipy.sparse?
        # We have Cover as dense numpy.
        # We can use:
        # indices = np.where(Cover[:, j] == 1)[0]
        # This is a column extraction. Numpy does this in C.
        # 3876 calls to np.where on a 50388-element array.
        # Should be fast enough (< 1s).
        
        idxs = np.where(Cover[:, j] == 1)[0]
        t_subset_blocks[j] = idxs.tolist()
        
    # Now perform swaps
    # We iterate through active blocks.
    # For each block, find its "unique" subsets (where cover_count == 1).
    # If it has no unique subsets, it's already removable (handled in deletion).
    # If it has unique subsets, we need a replacement that covers ALL of them.
    
    # To find a replacement quickly:
    # Let U be the set of unique subsets for block B.
    # We need a block B' such that U is a subset of the subsets covered by B'.
    # This is equivalent to: for all u in U, B' is in t_subset_blocks[u].
    # So B' must be in the intersection of t_subset_blocks[u] for all u in U.
    
    # Since |U| is small (usually 1 or 2), we can just check candidates from the first set.
    
    found_swap = True
    while found_swap and time.time() < swap_deadline:
        found_swap = False
        indices_list = list(active_blocks)
        random.shuffle(indices_list)
        
        for b_idx in indices_list:
            if b_idx not in active_blocks:
                continue
                
            # Find unique subsets
            # unique_mask = (Cover[b_idx] == 1) & (cover_counts == 1)
            # unique_indices = np.where(unique_mask)[0]
            unique_mask = (Cover[b_idx] == 1) & (cover_counts == 1)
            unique_indices = np.where(unique_mask)[0]
            
            if len(unique_indices) == 0:
                continue # Should have been removed in deletion phase, but maybe counts changed?
                
            # We need to find a replacement block b_new such that
            # b_new covers all t in unique_indices
            # AND b_new is not in active_blocks (or if it is, swapping is useless? No, if it's in active, we can't add it again. We need a NEW block or a block not currently in the set? Actually, if we swap, we remove b_idx and add b_new. If b_new is already in active, we just remove b_idx? No, that's deletion. Swap implies b_new is NOT in active_blocks, or we are replacing b_idx with b_new where b_new != b_idx. If b_new is already in active, adding it does nothing, removing b_idx is just deletion. So we require b_new NOT in active_blocks.)
            
            # Candidate generation:
            # Pick the first unique subset u0.
            # Any valid replacement MUST be in t_subset_blocks[u0].
            u0 = unique_indices[0]
            candidates = t_subset_blocks[u0]
            
            # Filter candidates:
            # 1. Not in active_blocks
            # 2. Covers all other unique subsets
            
            # To speed up, we can shuffle candidates and check a limited number, or check all if small.
            # t_subset_blocks[u0] size is roughly (number of blocks covering a specific 4-subset).
            # C(19-4, 7-4) = C(15, 3) = 455.
            # So ~455 candidates. Checking each against other unique subsets is fast.
            
            random.shuffle(candidates)
            
            for new_idx in candidates:
                if new_idx in active_blocks:
                    continue
                    
                # Check if new_idx covers all unique_indices
                valid = True
                for u in unique_indices[1:]: # We already know it covers u0
                    if new_idx not in t_subset_blocks[u]:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    active_blocks.remove(b_idx)
                    active_blocks.add(new_idx)
                    
                    # Update cover counts
                    cover_counts -= Cover[b_idx]
                    cover_counts += Cover[new_idx]
                    
                    found_swap = True
                    break # Restart the outer loop to allow for cascading deletions
                    
            if found_swap:
                break # Break outer loop to restart deletion check? 
                # Actually, after a swap, we should go back to deletion to clean up.
                # So we break out of the swap loop and let the main while loop handle it?
                # My structure is: Deletion -> Swap.
                # If we swap, we should probably run deletion again.
                # Let's structure it as a loop:
                # while time < limit:
                #   do deletion
                #   do one swap attempt
                #   if no swap, break
                
    # Refine the loop structure for better interaction between deletion and swap
    
    # Let's restart the optimization phase with a cleaner loop
    # Re-initialize cover_counts from scratch to be safe? No, it's maintained.
    # But the previous code structure was a bit messy. Let's rewrite the optimization part cleanly.
    
    # --- Refined Optimization ---
    
    # 1. Initial Deletion
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        indices_list = list(active_blocks)
        random.shuffle(indices_list)
        for idx in indices_list:
            if idx not in active_blocks: continue
            if not np.any((Cover[idx] == 1) & (cover_counts == 1)):
                active_blocks.remove(idx)
                cover_counts -= Cover[idx]
                changed = True

    # 2. Iterative Swap & Delete
    # We try to find a swap that reduces the "potential" for future blocks, or simply allows deletion.
    # Actually, a swap that maintains validity but allows immediate deletion of the swapped-out block's dependencies?
    # The standard heuristic is:
    # Pick a block B.
    # Find its unique subsets U.
    # Find a block B' that covers U.
    # Swap B for B'.
    # Then run deletion. If deletion removes B' or others, we improved.
    # If deletion removes nothing, the size is same. But maybe B' is "better" positioned?
    # For simplicity, we only accept a swap if it leads to a reduction in block count in the subsequent deletion pass.
    
    swap_limit = deadline - 2.0
    while time.time() < swap_limit:
        # Try to find a beneficial swap
        indices_list = list(active_blocks)
        random.shuffle(indices_list)
        
        swap_made = False
        for b_idx in indices_list:
            if b_idx not in active_blocks: continue
            
            unique_mask = (Cover[b_idx] == 1) & (cover_counts == 1)
            unique_indices = np.where(unique_mask)[0]
            
            if len(unique_indices) == 0:
                continue
                
            u0 = unique_indices[0]
            candidates = t_subset_blocks[u0]
            
            # Limit candidates to avoid long stalls
            # If there are too many, sample
            if len(candidates) > 100:
                candidates = random.sample(candidates, 100)
            else:
                candidates = list(candidates)
            
            random.shuffle(candidates)
            
            for new_idx in candidates:
                if new_idx in active_blocks:
                    continue
                    
                # Check coverage of all unique subsets
                valid = True
                for u in unique_indices[1:]:
                    if new_idx not in t_subset_blocks[u]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Tentative Swap
                active_blocks.remove(b_idx)
                active_blocks.add(new_idx)
                cover_counts -= Cover[b_idx]
                cover_counts += Cover[new_idx]
                
                # Try Deletion
                initial_size = len(active_blocks)
                changed = True
                while changed and time.time() < swap_limit - 0.5:
                    changed = False
                    del_indices = list(active_blocks)
                    random.shuffle(del_indices)
                    for idx in del_indices:
                        if idx not in active_blocks: continue
                        if not np.any((Cover[idx] == 1) & (cover_counts == 1)):
                            active_blocks.remove(idx)
                            cover_counts -= Cover[idx]
                            changed = True
                            
                final_size = len(active_blocks)
                
                if final_size < initial_size:
                    # Swap was beneficial (allowed deletion)
                    swap_made = True
                    break # Accept swap and break out to try more swaps/deletions
                else:
                    # Revert swap
                    active_blocks.remove(new_idx)
                    active_blocks.add(b_idx)
                    cover_counts -= Cover[new_idx]
                    cover_counts += Cover[b_idx]
            
            if swap_made:
                break
        
        if not swap_made:
            break # No more beneficial swaps found
            
    # Final Deletion Pass
    if time.time() < deadline - 0.5:
        changed = True
        while changed and time.time() < deadline - 0.5:
            changed = False
            indices_list = list(active_blocks)
            random.shuffle(indices_list)
            for idx in indices_list:
                if idx not in active_blocks: continue
                if not np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    active_blocks.remove(idx)
                    cover_counts -= Cover[idx]
                    changed = True

    result = [list(all_kblocks[i]) for i in active_blocks]
    return result
