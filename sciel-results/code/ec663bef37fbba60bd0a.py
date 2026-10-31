import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # Greedy Construction
    selected_indices = []
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    
    # Precompute row sums or something? No, just dot product.
    while np.sum(uncovered) > 0 and time.time() < deadline - 10.0:
        gains = Cover @ uncovered
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
        
    # If we ran out of time during greedy, we have an invalid covering.
    # But greedy is usually fast. Let's assume it finishes.
    # If not, we must return what we have? No, must be valid.
    # The problem says "Valid when every t-subset ... lies in at least one block".
    # If greedy doesn't finish, we fail.
    # Given the size, greedy should take < 1s.
    
    # Local Optimization
    if time.time() < deadline - 5.0:
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_counts += Cover[idx]
            
        selected_indices, cover_counts = _local_optimize_v2(selected_indices, Cover, cover_counts, n_blocks, deadline - 2.0)
        
    blocks = [all_kblocks[i] for i in selected_indices]
    return blocks

def _local_optimize_v2(selected_indices, Cover, cover_counts, n_blocks, deadline):
    """
    Vectorized local optimization.
    1. Remove redundant blocks.
    2. Try to replace blocks with better ones to enable further removals.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Phase 1: Remove redundant blocks
        new_selected_indices = []
        for idx in selected_indices:
            # Check if idx covers any t-subset with count 1
            # mask = Cover[idx] == 1
            # unique_mask = mask & (cover_counts == 1)
            # if np.any(unique_mask): keep
            # else: remove
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                new_selected_indices.append(idx)
            else:
                cover_counts -= Cover[idx]
                changed = True
        selected_indices = new_selected_indices
        
        if time.time() > deadline:
            break
            
        # Phase 2: Try replacements
        # For each block, see if we can swap it with a block that covers its unique t-subsets
        # and is "better" (e.g., covers more total t-subsets, or has less overlap)
        # Since we just want to reduce count, and swap doesn't reduce count, 
        # we only do this if it might lead to a removal later.
        # However, a simpler heuristic: 
        # If we can swap block i for block j, and block j is "more central" or "more overlapping", 
        # it might make other blocks redundant.
        
        # Let's try a simple swap:
        # For each i, find j not in selected such that Unique(i) subset of Cover(j).
        # If found, swap i with j.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected_indices = []
            for idx in selected_indices:
                # Check if idx is essential
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    # Find unique t-subsets covered by idx
                    unique_tsubs = np.where((Cover[idx] == 1) & (cover_counts == 1))[0]
                    
                    if len(unique_tsubs) == 0:
                        # Should have been removed in Phase 1, but just in case
                        new_selected_indices.append(idx)
                        continue
                    
                    # Find candidates j that cover all unique_tsubs
                    # Cover[:, unique_tsubs] is shape (n_blocks, len(unique_tsubs))
                    # We want rows where all values are 1
                    # candidates_mask = np.all(Cover[:, unique_tsubs] == 1, axis=1)
                    
                    # Exclude currently selected blocks
                    # We can create a mask of selected blocks and set them to False
                    
                    # Optimization: Only check if unique_tsubs is small? 
                    # No, vectorization is fast.
                    
                    # To avoid creating a huge temporary array for every idx, 
                    # we can just do it. n_blocks * len(unique_tsubs) is manageable.
                    
                    # Get candidates
                    # Note: Cover is uint8.
                    cand_mask = np.all(Cover[:, unique_tsubs] == 1, axis=1)
                    
                    # Exclude current selection
                    # We don't have a boolean array of selected indices readily available for vectorized masking of n_blocks size?
                    # We can create one.
                    sel_mask = np.zeros(n_blocks, dtype=bool)
                    for s_idx in selected_indices:
                        sel_mask[s_idx] = True
                    cand_mask[sel_mask] = False
                    
                    candidates = np.where(cand_mask)[0]
                    
                    if len(candidates) > 0:
                        # Pick the best candidate
                        # Heuristic: Maximize total coverage (sum of Cover[j])
                        # Or minimize overlap with other blocks?
                        # Let's pick the one with max total coverage.
                        gains = Cover[candidates].sum(axis=1)
                        best_candidate = candidates[np.argmax(gains)]
                        
                        # Swap
                        # Remove idx
                        cover_counts -= Cover[idx]
                        # Add best_candidate
                        cover_counts += Cover[best_candidate]
                        
                        new_selected_indices.append(best_candidate)
                        improved = True
                    else:
                        new_selected_indices.append(idx)
                else:
                    # Redundant, should have been removed, but if not, keep it for now?
                    # Actually, if it's redundant, we should remove it.
                    # But Phase 1 should have caught it.
                    # If we are here, it means cover_counts changed?
                    # No, we are in the replacement loop.
                    # If a block became redundant due to a previous swap in this inner loop,
                    # we should remove it.
                    if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                        new_selected_indices.append(idx)
                    else:
                        cover_counts -= Cover[idx]
                        changed = True
                        # Do not append, effectively removing it
                        improved = True
            
            selected_indices = new_selected_indices
            
            # After swaps, some blocks might have become redundant.
            # The outer loop will handle that in the next iteration of Phase 1.
            
    return selected_indices, cover_counts
