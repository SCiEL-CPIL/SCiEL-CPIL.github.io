import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}
    
    # Precompute all k-blocks and their coverage masks
    # For v=24, k=7, there are 346,104 blocks.
    # Each block covers C(7,3)=35 t-subsets.
    # We store:
    # 1. list of all block tuples
    # 2. list of coverage masks (integers)
    # 3. map from block tuple to index
    # 4. inverse index: t-subset index -> list of block indices covering it
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    # Inverse index: for each t-subset, which blocks cover it?
    # Using lists of block indices
    inv_index = [[] for _ in range(n_tsubs)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_index[tsub]
            mask |= (1 << idx)
            inv_index[idx].append(i)
        block_masks[i] = mask
        
    full_mask = (1 << n_tsubs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to get a valid covering from a set of block indices
    def get_valid_covering(block_indices):
        # Verify coverage
        covered = 0
        for idx in block_indices:
            covered |= block_masks[idx]
        if covered != full_mask:
            return None
        # Remove duplicates (shouldn't happen if we use a set, but good for safety)
        unique_indices = list(set(block_indices))
        return [list(all_blocks[i]) for i in unique_indices]

    # Helper: Greedy construction
    def greedy_construct(rng):
        covered = 0
        selected = []
        
        # To speed up greedy, we can pick the block with max new coverage.
        # Naive O(n_blocks * n_tsubs) per step is too slow.
        # Instead: Pick a random uncovered t-subset, pick a random block covering it?
        # Or: Pick the block that covers the most uncovered t-subsets.
        
        # Efficient Greedy:
        # 1. Find an uncovered t-subset.
        # 2. Among blocks covering it, pick the one that covers the most *other* uncovered t-subsets.
        
        while covered != full_mask:
            if time.time() > deadline - 2.0:
                return None
            
            # Find first uncovered t-subset
            uncovered_mask = full_mask ^ covered
            # Get index of least significant set bit
            lsb = uncovered_mask & (-uncovered_mask)
            focus_idx = lsb.bit_length() - 1
            
            # Get candidates covering focus_idx
            candidates = inv_index[focus_idx]
            
            best_candidate = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = bin(block_masks[b_idx] & uncovered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx
            
            if best_candidate == -1:
                return None # Should not happen if problem is solvable
                
            selected.append(best_candidate)
            covered |= block_masks[best_candidate]
            
        return selected

    # Helper: Local Search (1-Exchange + Redundancy Removal)
    def local_search(block_indices, rng):
        # Convert to set for O(1) lookup
        current_set = set(block_indices)
        covered = 0
        for idx in current_set:
            covered |= block_masks[idx]
            
        if covered != full_mask:
            return list(current_set)
            
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # 1. Remove Redundant Blocks
            # A block is redundant if all its t-subsets are covered by others.
            # We can check this by: (covered_without_this_block) == covered
            # Or: (mask_of_block & (covered ^ mask_of_block)) == 0 ? No.
            # If (mask_of_block & (covered ^ mask_of_block)) == 0, it means every bit in mask_of_block 
            # is also in (covered ^ mask_of_block)? No.
            # covered ^ mask_of_block is the coverage if we remove this block.
            # We want covered_without == covered.
            # i.e., (covered & ~mask_of_block) == (covered ^ mask_of_block)?
            # Actually, simply: if (covered ^ block_masks[idx]) == covered, then removing it changes nothing.
            
            to_remove = []
            for idx in list(current_set):
                if (covered ^ block_masks[idx]) == covered:
                    to_remove.append(idx)
                    
            if to_remove:
                for idx in to_remove:
                    current_set.remove(idx)
                    covered ^= block_masks[idx] # XOR works because bits are 1 in both
                improved = True
                continue # Restart loop as coverage changed significantly
            
            # 2. 1-Exchange: Try to replace a block with another to enable further reduction
            # For each block B, find unique t-subsets (covered only by B).
            # If none, it's redundant (handled above).
            # If some, find a block B' that covers all unique t-subsets.
            # If B' exists and is not in set, swap B with B'.
            
            swaps_made = False
            for idx in list(current_set):
                if time.time() > deadline - 2.0:
                    break
                
                mask_b = block_masks[idx]
                # Unique t-subsets for B: bits in mask_b that are NOT in (covered ^ mask_b)
                # covered_without_b = covered ^ mask_b
                # unique = mask_b & ~(covered_without_b)
                covered_without_b = covered ^ mask_b
                unique_mask = mask_b & (~covered_without_b)
                
                if unique_mask == 0:
                    continue # Redundant, should have been caught above
                
                # Find a block B' that covers unique_mask
                # Get list of t-subset indices in unique_mask
                unique_indices = []
                m = unique_mask
                while m:
                    lsb = m & (-m)
                    unique_indices.append(lsb.bit_length() - 1)
                    m ^= lsb
                
                # Intersect inv_index for these t-subsets
                # Start with list of blocks covering the first unique t-subset
                candidates = set(inv_index[unique_indices[0]])
                
                for u_idx in unique_indices[1:]:
                    candidates &= set(inv_index[u_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                    
                # Check if any candidate is NOT in current_set
                for new_idx in candidates:
                    if new_idx not in current_set:
                        # Perform swap
                        current_set.remove(idx)
                        current_set.add(new_idx)
                        covered = (covered ^ mask_b) | block_masks[new_idx]
                        swaps_made = True
                        improved = True
                        break # Restart local search loop
                        
                if swaps_made:
                    break # Restart loop
                    
        return list(current_set)

    # Main Loop: Multiple Restarts
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        rng = random.Random(restart * 12345 + 6789)
        
        # 1. Greedy
        init_blocks = greedy_construct(rng)
        if not init_blocks:
            continue
            
        # 2. Local Search
        optimized_blocks = local_search(init_blocks, rng)
        
        # 3. Validate and Update Best
        if optimized_blocks:
            final_covering = get_valid_covering(optimized_blocks)
            if final_covering:
                count = len(final_covering)
                if count < best_count:
                    best_count = count
                    best_blocks = final_covering
                    # Optional: If we hit a very good number, we can stop early?
                    # Lower bound is 69. Best known 76.
                    # If we get 76, that's a tie with known best.
                    # If we get < 76, that's a breakthrough.
                    if best_count <= 75:
                        # We found something better than best known! 
                        # But we should still try to improve or just return quickly?
                        # Let's allow a bit more time to see if we can go lower.
                        pass

    # Fallback if no valid covering was found (should not happen)
    if best_blocks is None:
        # Simple greedy fallback
        covered = 0
        blocks = []
        rng = random.Random(42)
        while covered != full_mask:
            uncovered_mask = full_mask ^ covered
            lsb = uncovered_mask & (-uncovered_mask)
            focus_idx = lsb.bit_length() - 1
            candidates = inv_index[focus_idx]
            best_candidate = -1
            best_gain = -1
            for b_idx in candidates:
                gain = bin(block_masks[b_idx] & uncovered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx
            if best_candidate == -1:
                break
            blocks.append(list(all_blocks[best_candidate]))
            covered |= block_masks[best_candidate]
        best_blocks = blocks
        best_count = len(blocks)

    return best_blocks
