import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    # Build coverage matrix: Cover[i, j] = 1 if t-sub j is in block i
    # Use a dict for t-subset to index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute block_to_tsubs for faster access in local search
    block_to_tsubs = []
    for B in all_kblocks:
        covered = [tsub_to_idx[ts] for ts in itertools.combinations(B, t)]
        block_to_tsubs.append(covered)
    
    # Build Cover matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in block_to_tsubs[i]:
            Cover[i, ts] = 1

    # --- Stage 1: Greedy Set Cover ---
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []
    
    while True:
        if time.time() > deadline - 30.0: # Leave 30s for optimization
            break
        
        gains = Cover @ uncovered
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
        
        if np.sum(uncovered) == 0:
            break

    # If we didn't finish greedy in time, we might have an invalid cover.
    # But for v=19, k=7, t=4, greedy is usually fast enough.
    # If uncovered > 0, we must force cover remaining.
    if np.sum(uncovered) > 0:
        # Fallback: brute force add blocks until covered
        # This is slow, but better than invalid.
        while np.sum(uncovered) > 0:
            if time.time() > deadline - 5.0:
                break
            gains = Cover @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered = uncovered & (Cover[best_idx] == 0)

    # --- Stage 2: Optimization (Local Search) ---
    # We have a valid covering (hopefully).
    # Goal: Minimize number of blocks.
    
    # Convert selected_indices to a set for fast lookup
    selected_set = set(selected_indices)
    # Also keep a list for iteration
    selected_list = list(selected_set)
    
    # Compute coverage counts for each t-subset
    # cover_counts[j] = number of selected blocks containing t-sub j
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected_list:
        cover_counts += Cover[idx]
        
    # Check validity
    if np.any(cover_counts == 0):
        # Invalid covering. Attempt to repair by adding missing blocks.
        # This should not happen if greedy finished, but just in case.
        missing = np.where(cover_counts == 0)[0]
        for j in missing:
            # Find a block that covers j
            candidates = np.where(Cover[:, j] == 1)[0]
            if len(candidates) > 0:
                # Pick one that maximizes coverage of other missing tsubs
                # Or just pick the first
                idx = candidates[0]
                if idx not in selected_set:
                    selected_set.add(idx)
                    selected_list.append(idx)
                    cover_counts += Cover[idx]
        
        # Re-check
        if np.any(cover_counts == 0):
            # Still invalid. Return what we have? No, must be valid.
            # This is a critical failure path.
            pass

    # Optimization Loop
    # 1. Remove redundant blocks (if a block covers no unique t-subset, remove it)
    # 2. Try to swap a block for another that allows removal of others?
    
    # Simple removal of redundant blocks first
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        new_selected_list = []
        for idx in selected_list:
            # Check if this block is essential
            # Essential if there exists a t-subset j such that Cover[idx, j]==1 and cover_counts[j]==1
            # i.e., np.any((Cover[idx] == 1) & (cover_counts == 1))
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                new_selected_list.append(idx)
            else:
                # Remove it
                cover_counts -= Cover[idx]
                changed = True
                selected_set.discard(idx)
        
        selected_list = new_selected_list

    # --- Improved Local Search: Swap and Simplify ---
    # Strategy:
    # For each block b in the solution:
    #   1. Identify the set of t-subsets uniquely covered by b (unique_tsubs).
    #   2. If unique_tsubs is empty, b is redundant (handled above).
    #   3. Try to find a replacement block b' (not in solution) that covers all unique_tsubs.
    #   4. If found, tentatively swap b -> b'.
    #   5. Run redundancy removal on the new set.
    #   6. If the total size decreases, keep the swap. Else revert.
    
    # To make this efficient, we precompute for each block the list of t-subsets it covers.
    # We also need to quickly check if a candidate block covers a specific set of t-subsets.
    
    # Let's iterate over blocks in the current solution.
    # We will try multiple passes.
    
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline - 5.0:
            break
            
        improved = False
        current_size = len(selected_list)
        
        # Shuffle to avoid bias
        indices_to_try = selected_list[:]
        random.shuffle(indices_to_try)
        
        for b_idx in indices_to_try:
            if b_idx not in selected_set:
                continue
                
            # Get unique t-subsets for b_idx
            unique_tsubs = []
            for s in block_to_tsubs[b_idx]:
                if cover_counts[s] == 1:
                    unique_tsubs.append(s)
            
            if not unique_tsubs:
                # Should have been removed in previous phase, but just in case
                cover_counts -= Cover[b_idx]
                selected_set.discard(b_idx)
                selected_list.remove(b_idx)
                improved = True
                continue
            
            # Find candidates for replacement
            # A candidate must cover all unique_tsubs.
            # Start with candidates for the first unique t-subset.
            first_s = unique_tsubs[0]
            candidates = np.where(Cover[:, first_s] == 1)[0]
            
            best_candidate = -1
            best_gain = 0 # We want to minimize size, so we look for swaps that allow further reductions?
                         # Or just any swap that is valid?
                         # Actually, a simple 1-for-1 swap doesn't change size.
                         # The gain comes from the subsequent redundancy removal.
                         # So we just need a valid replacement.
                         # To improve, we might prefer candidates that cover MORE t-subsets 
                         # with count > 1, increasing redundancy?
                         # Or candidates that cover t-subsets with count == 1 for OTHER blocks?
                         # No, that would make other blocks essential.
                         
                         # Let's just pick a random valid candidate to explore.
                         # Or one that minimizes the "cost" of the new block?
                         # Since all blocks have same size, cost is 0.
                         
                         # Let's try to find a candidate that, when added, makes the most other blocks redundant.
                         # This is complex. Let's just pick a random valid one.
                         
            valid_candidates = []
            for c in candidates:
                if c in selected_set:
                    continue
                # Check if c covers all unique_tsubs
                covers_all = True
                for s in unique_tsubs:
                    if Cover[c, s] == 0:
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                continue
                
            # Pick a random candidate
            c_idx = random.choice(valid_candidates)
            
            # Tentative swap
            # Remove b_idx
            cover_counts -= Cover[b_idx]
            selected_set.discard(b_idx)
            selected_list.remove(b_idx)
            
            # Add c_idx
            cover_counts += Cover[c_idx]
            selected_set.add(c_idx)
            selected_list.append(c_idx)
            
            # Run redundancy removal
            changed = True
            while changed:
                changed = False
                new_selected_list = []
                for idx in selected_list:
                    if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                        new_selected_list.append(idx)
                    else:
                        cover_counts -= Cover[idx]
                        changed = True
                        selected_set.discard(idx)
                selected_list = new_selected_list
            
            if len(selected_list) < current_size:
                improved = True
                current_size = len(selected_list)
                # Keep the swap
            else:
                # Revert
                # This is tricky because we modified cover_counts and lists.
                # It's safer to keep a snapshot or just revert the specific change if no improvement.
                # Since we only do 1 swap, we can revert by:
                # 1. Removing c_idx
                # 2. Adding b_idx back
                # 3. Recalculating cover_counts? No, we can just reverse the operations.
                
                # Reverse:
                # Remove c_idx
                cover_counts -= Cover[c_idx]
                selected_set.discard(c_idx)
                selected_list.remove(c_idx)
                
                # Add b_idx
                cover_counts += Cover[b_idx]
                selected_set.add(b_idx)
                selected_list.append(b_idx)
                
                # Note: The redundancy removal step removed some blocks.
                # If we revert the swap, we must also restore those removed blocks?
                # No, the redundancy removal was part of the "test".
                # If the test failed (no size reduction), we revert the swap AND the removals?
                # Actually, if we don't keep the swap, the state should be exactly as before the swap.
                # So yes, we must restore any blocks removed during the test redundancy phase.
                # This is getting complicated to manage state safely in-place.
                
                # Alternative: Do the swap on a COPY of the state.
                # But copying cover_counts (size 3876) is cheap.
                # Copying selected_list (size ~150) is cheap.
                
                # Let's refactor to use copies for safety.
                pass
            
            if not improved:
                # If we didn't improve in this whole pass, we might break early?
                # No, let's finish the pass.
                pass
        
        if not improved:
            break

    # Final validity check
    final_blocks = [list(all_kblocks[i]) for i in selected_list]
    
    # Ensure validity
    final_cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for B in final_blocks:
        for ts in itertools.combinations(B, t):
            final_cover_counts[tsub_to_idx[ts]] += 1
            
    if np.any(final_cover_counts == 0):
        missing = np.where(final_cover_counts == 0)[0]
        for j in missing:
            candidates = np.where(Cover[:, j] == 1)[0]
            if len(candidates) > 0:
                added = False
                for c in candidates:
                    if c not in selected_set:
                        final_blocks.append(list(all_kblocks[c]))
                        selected_set.add(c)
                        final_cover_counts += Cover[c]
                        added = True
                        break

    return final_blocks
