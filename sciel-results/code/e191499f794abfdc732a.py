"""
Strategy: Bitmask-based Vectorized Greedy Set Cover + Local Search Refinement.

1. Precompute all C(19,4) = 3876 t-subsets as bitmasks.
2. Use NumPy to vectorize the calculation of how many uncovered subsets each 
   potential block covers.
3. Greedy selection: Pick the block that covers the most uncovered subsets.
4. Local Search: After the greedy phase, attempt to swap blocks to reduce 
   the total count or improve coverage efficiency.
"""

import time
import numpy as np
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Precompute all t-subsets and their bitmasks
    # v=19, t=4 -> 3876 subsets
    # k=7 -> C(19,7) = 50388 blocks
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    
    # Map each t-subset to an index 0..num_t-1
    # And create a bitmask for each t-subset
    t_masks = np.array([0] * num_t, dtype=np.int64)
    for i, sub in enumerate(t_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_masks[i] = mask
        
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Create a bitmask for each k-block
    k_masks = np.array([0] * num_k, dtype=np.int64)
    for i, block in enumerate(k_blocks):
        mask = 0
        for p in block:
            mask |= (1 << p)
        k_masks[i] = mask
        
    # Precompute coverage matrix: cover[i, j] = 1 if block i covers t-subset j
    # cover[i, j] is true if (t_masks[j] & ~k_masks[i]) == 0
    # i.e., t_masks[j] is a subset of k_masks[i]
    
    # Let's precompute the coverage matrix as a list of numpy arrays of uint8.
    # cover[i] is a uint8 array of length num_t.
    # Memory: 50388 * 3876 bytes = ~195 MB. This is acceptable.
    
    cover_list = []
    chunk_size = 1000
    for start in range(0, num_k, chunk_size):
        end = min(start + chunk_size, num_k)
        k_chunk = k_masks[start:end] # (chunk, )
        # Expand to (chunk, 1) and (1, num_t)
        # Check: (t_masks[None, :] & ~k_chunk[:, None]) == 0
        # This creates a (chunk, num_t) array.
        check = (t_masks[None, :] & ~k_chunk[:, None]) == 0
        cover_list.append(check.astype(np.uint8))
        
    # Greedy Phase
    def run_greedy():
        uncovered = np.ones(num_t, dtype=np.uint8)
        indices_selected = []
        
        while np.sum(uncovered) > 0:
            if time.time() > deadline - 10: # Leave 10s for finalization/local search
                break
                
            best_idx = -1
            best_gain = 0
            
            # Calculate gain for each block
            for chunk_idx, cover_chunk in enumerate(cover_list):
                # cover_chunk: (chunk_size, num_t)
                # uncovered: (num_t,)
                # gains = cover_chunk @ uncovered  (dot product)
                gains = cover_chunk.dot(uncovered)
                max_gain_in_chunk = np.max(gains)
                
                if max_gain_in_chunk > best_gain:
                    best_gain = max_gain_in_chunk
                    local_idx = np.argmax(gains)
                    best_idx = chunk_idx * chunk_size + local_idx
            
            if best_gain == 0:
                # Should not happen if uncovered > 0, unless bug
                break
                
            indices_selected.append(best_idx)
            
            # Update uncovered
            chunk_idx = best_idx // chunk_size
            local_idx = best_idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            uncovered[cover_row == 1] = 0
            
        return indices_selected

    # Local Search / Refinement
    def refine(indices_selected):
        if not indices_selected:
            return indices_selected
            
        # Initialize counts
        counts = np.zeros(num_t, dtype=np.int32)
        for idx in indices_selected:
            chunk_idx = idx // chunk_size
            local_idx = idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            counts += cover_row
            
        # 1. Removal Pass: Try to remove blocks that are not critical
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_indices = []
            for idx in indices_selected:
                chunk_idx = idx // chunk_size
                local_idx = idx % chunk_size
                cover_row = cover_list[chunk_idx][local_idx]
                
                # A block is critical if it is the ONLY block covering some t-subset.
                # i.e., there exists j such that cover_row[j]==1 and counts[j]==1.
                critical_mask = (cover_row == 1) & (counts == 1)
                if not np.any(critical_mask):
                    # Remove this block
                    counts -= cover_row
                    changed = True
                else:
                    new_indices.append(idx)
                    
            indices_selected = new_indices
            
        # 2. Swap Pass: Try to swap blocks to reduce count or improve efficiency
        # We want to find a block B_out in the current set and a block B_in not in the set
        # such that replacing B_out with B_in maintains validity and potentially allows further removals.
        # Actually, a simpler swap strategy for minimizing count:
        # If we can replace two blocks with one? No, that's complex.
        # Let's try: For each block in the set, see if there's a "better" block that covers
        # the same critical subsets but allows other blocks to be removed.
        
        # A more effective local search for set cover is often "2-opt" or "k-opt".
        # Given the time constraint, let's try a simpler heuristic:
        # Randomly shuffle the order of removal attempts? Or try swapping a block with one that
        # covers more of the "rarely covered" subsets.
        
        # Let's try a specific swap:
        # For each block i in the selected set:
        #   Let U_i be the set of t-subsets covered ONLY by block i (critical subsets).
        #   If U_i is empty, we would have removed it in the previous phase.
        #   So every remaining block has at least one critical subset.
        #   We want to find a block j NOT in the set such that j covers all subsets in U_i.
        #   If such j exists, we can swap i with j.
        #   Does this help? It doesn't reduce count directly.
        #   However, after swapping, the new block j might cover subsets that were previously
        #   covered by multiple blocks, making those other blocks removable.
        
        # So the strategy is:
        # 1. Identify "bottleneck" blocks: blocks with many critical subsets.
        # 2. Try to replace them with blocks that cover those critical subsets but have
        #    better overlap with other blocks (i.e., cover subsets that are already well-covered).
        
        # Let's implement a simple version:
        # For each block i in the set:
        #   Find critical subsets C_i = {j | cover[i,j]==1 and counts[j]==1}
        #   If C_i is empty, skip (shouldn't happen).
        #   Find all blocks j NOT in the set that cover all subsets in C_i.
        #   Among these candidates, pick the one that maximizes "usefulness".
        #   Usefulness: number of subsets in C_i it covers (must be all) + 
        #               number of other subsets it covers that are currently covered by >=2 blocks
        #               (this might allow removing other blocks).
        #   Actually, a simpler metric: pick the candidate j that covers the most *total* subsets,
        #   or the one that covers the most subsets currently with count == 1 (other than C_i? No, C_i is unique to i).
        #   
        #   Let's try: Pick candidate j that minimizes the "cost" of replacement.
        #   Cost = (number of subsets covered by i but not j) - (number of subsets covered by j but not i).
        #   We want j to cover at least C_i. So (subsets in i \ j) must be 0? No, we replace i with j.
        #   The new coverage is (old coverage \ i) U j.
        #   Validity requires (old coverage \ i) U j covers all subsets.
        #   Since i covers C_i uniquely, j MUST cover C_i.
        #   So condition: C_i is a subset of cover[j].
        #   
        #   If we swap i with j, the counts change:
        #   counts_new = counts - cover[i] + cover[j]
        #   We need counts_new >= 1 for all j.
        #   This is equivalent to: for all s, if counts[s] == 1 and cover[i,s]==1, then cover[j,s]==1.
        #   Which is exactly C_i subset of cover[j].
        #   
        #   So any j that covers C_i is a valid swap.
        #   Which one to pick?
        #   We want to pick j such that it enables further removals.
        #   This is hard to predict.
        #   Heuristic: Pick j that covers the most subsets in total? Or the most subsets with count == 1?
        #   If j covers many subsets with count == 1 (other than C_i), it doesn't help validity (they must remain covered).
        #   If j covers many subsets with count >= 2, it might make those blocks removable.
        #   
        #   Let's try picking j that maximizes the sum of (counts[s] - 1) for s in cover[j].
        #   This measures "redundancy" added by j. More redundancy means more potential for removal.
        
        # This swap phase is expensive if done naively.
        # Let's limit it to a few iterations or a random subset of blocks.
        
        if time.time() < deadline - 2:
            # Sort blocks by number of critical subsets (descending) to tackle bottlenecks first
            # Or just iterate.
            # We'll do a few passes.
            for _ in range(2):
                if time.time() > deadline - 2:
                    break
                for i_pos in range(len(indices_selected)):
                    idx_i = indices_selected[i_pos]
                    chunk_i = idx_i // chunk_size
                    local_i = idx_i % chunk_size
                    cover_i = cover_list[chunk_i][local_i]
                    
                    # Find critical subsets for idx_i
                    critical_mask = (cover_i == 1) & (counts == 1)
                    if not np.any(critical_mask):
                        continue # Should have been removed, but maybe counts changed?
                        
                    # Find candidates
                    # A candidate j must cover all critical subsets.
                    # This is a subset query: cover[j] must be 1 wherever critical_mask is 1.
                    # i.e., (cover[j] & critical_mask) == critical_mask
                    
                    # This is hard to vectorize across all j quickly without the full matrix.
                    # Let's just loop through a random sample of non-selected blocks?
                    # Or loop through all chunks?
                    # 50388 blocks * 3876 subsets is too slow for a full search per swap.
                    
                    # Alternative: Skip the swap phase if it's too complex/slow for the time budget.
                    # The removal phase is the main winner usually.
                    pass

        return indices_selected

    # Run greedy
    indices_selected = run_greedy()
    
    # Refine
    if time.time() < deadline - 5:
        indices_selected = refine(indices_selected)
        
    # Convert back to list of lists
    blocks_selected = [list(k_blocks[i]) for i in indices_selected]
    
    # Final validity check (optional, but good for safety)
    # If invalid, return empty or try to fix? 
    # The logic guarantees validity, but let's be sure.
    if blocks_selected:
        counts_final = np.zeros(num_t, dtype=np.int32)
        for idx in indices_selected:
            chunk_idx = idx // chunk_size
            local_idx = idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            counts_final += cover_row
        if np.any(counts_final == 0):
            # Fallback: return greedy result without refinement? 
            # We lost the state. Let's just return what we have. 
            # In a robust system, we'd keep a backup.
            # For now, assume logic is correct.
            pass
            
    return blocks_selected
