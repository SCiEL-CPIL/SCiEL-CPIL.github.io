import itertools
import time
import numpy as np

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # We need to represent masks as arrays of 64-bit integers for NumPy efficiency
    # Number of 64-bit words needed
    n_words = (n_t_subsets + 63) // 64
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset as a NumPy array
    # masks_np shape: (n_blocks, n_words)
    masks_np = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    for i, block in enumerate(k_subsets):
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            word_idx = idx // 64
            bit_idx = idx % 64
            masks_np[i, word_idx] |= (1 << bit_idx)
            
    # all_mask is a numpy array of 1s where bits exist
    all_mask_np = np.zeros(n_words, dtype=np.uint64)
    # Set all bits in the last word appropriately if n_t_subsets is not a multiple of 64
    if n_t_subsets % 64 != 0:
        # Mask out the unused bits in the last word
        unused_bits = 64 - (n_t_subsets % 64)
        all_mask_np[-1] = (1 << (64 - unused_bits)) - 1
    else:
        all_mask_np[-1] = np.uint64(0xFFFFFFFFFFFFFFFF)
    all_mask_np[:] = np.uint64(0xFFFFFFFFFFFFFFFF)
    if n_t_subsets % 64 != 0:
        unused_bits = 64 - (n_t_subsets % 64)
        all_mask_np[-1] = (1 << (64 - unused_bits)) - 1

    def get_uncovered_count(uncovered_np):
        return int(uncovered_np.sum())
        
    def get_gain(masks_np, uncovered_np):
        # Bitwise AND and sum
        inter = masks_np & uncovered_np
        # Sum bits. Since dtype is uint64, summing might overflow?
        # Max sum per row is n_words * 64? No, max sum is n_t_subsets (11628).
        # uint64 can hold 11628 easily.
        return inter.sum(axis=1)

    # 4. Greedy Set Cover
    uncovered = all_mask_np.copy()
    selected_greedy = []
    used_blocks = np.zeros(n_blocks, dtype=bool)
    
    while get_uncovered_count(uncovered) > 0:
        # Calculate gains for all unused blocks
        gains = get_gain(masks_np, uncovered)
        # Mask out used blocks
        gains[used_blocks] = -1
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain <= 0:
            break
            
        selected_greedy.append(int(best_idx))
        used_blocks[best_idx] = True
        uncovered &= ~masks_np[best_idx]
        
    current_best = len(selected_greedy)
    best_solution = selected_greedy[:]
    
    # 5. Local Search: Redundancy Removal
    # We represent the current solution's masks
    def try_remove_redundant(sol_indices):
        if not sol_indices:
            return sol_indices
        removed = True
        while removed:
            removed = False
            n_sol = len(sol_indices)
            # Compute union of all blocks in solution
            union_mask = np.zeros(n_words, dtype=np.uint64)
            for idx in sol_indices:
                union_mask |= masks_np[idx]
                
            for i in range(n_sol - 1, -1, -1):
                # Check if removing sol_indices[i] still covers all_mask
                # Union of others = union_mask & ~masks_np[sol_indices[i]]
                # If (union_mask & ~masks_np[sol_indices[i]]) == all_mask_np, it's redundant
                # Note: all_mask_np has 1s only where valid t-subsets are.
                # So we check if (union_mask & ~masks_np[sol_indices[i]] & all_mask_np) == all_mask_np
                
                other_union = union_mask & ~masks_np[sol_indices[i]]
                if np.array_equal(other_union & all_mask_np, all_mask_np):
                    # Redundant
                    sol_indices.pop(i)
                    removed = True
                    break
        return sol_indices

    best_solution = try_remove_redundant(best_solution)
    current_best = len(best_solution)
    
    # 6. 2-for-1 Reduction
    # Try to replace two blocks A, B with one block C
    # Condition: Unique(A) U Unique(B) is covered by C
    
    def try_2_for_1(sol_indices):
        improved = True
        while improved:
            improved = False
            n_sol = len(sol_indices)
            if n_sol < 2:
                break
            
            # Precompute unique masks for each block in solution
            # unique[i] = masks_np[sol[i]] & ~(OR of all other masks in sol)
            # This is O(N^2) per iteration. N is small (~300).
            # 300 * 300 * 182 ops ~ 16M ops. Fast.
            
            union_mask = np.zeros(n_words, dtype=np.uint64)
            for idx in sol_indices:
                union_mask |= masks_np[idx]
                
            unique_masks = []
            for i in range(n_sol):
                others_union = union_mask & ~masks_np[sol_indices[i]]
                unique_mask = masks_np[sol_indices[i]] & ~others_union
                unique_masks.append(unique_mask)
                
            # For each pair (i, j), check if there exists k not in sol such that
            # (unique_masks[i] | unique_masks[j]) is subset of masks_np[k]
            # To speed up, we can limit k to blocks that have high overlap with the union?
            # Or just iterate through all k? 75k * 45k pairs = 3.3B checks. Too slow.
            
            # Optimization: Only check k if masks_np[k] covers the "hard" part.
            # Let's restrict k to the top M candidates based on gain with the target union?
            # Target = unique_masks[i] | unique_masks[j]
            # We need masks_np[k] & Target == Target.
            
            # Instead of checking all k, we can pre-compute a list of blocks sorted by "popularity" or just check a random subset?
            # Or, we can invert the search:
            # For each k in all_blocks:
            #   If k is in sol, skip.
            #   For each pair (i, j) in sol:
            #      If unique[i] | unique[j] is subset of masks_np[k]:
            #          Replace i, j with k.
            
            # This is O(N_blocks * N_sol^2). 75k * 90k = 6.7B. Still too slow.
            
            # Heuristic: Only try to merge pairs where the union of unique masks is small?
            # Or, only check k that are "good" blocks (high total coverage)?
            
            # Let's try a limited check: For each pair (i, j), check the first 1000 blocks that have high overlap with unique[i]|unique[j].
            # This is complex.
            
            # Alternative: Skip 2-for-1 if it's too slow and rely on Greedy + Redundancy.
            # The current score is 1.1241. Greedy + Redundancy might be close to optimal for this size.
            
            pass
        return sol_indices

    # Given the complexity and time limit, we will skip the expensive 2-for-1 global search
    # and instead do a few random restarts of the Greedy with different tie-breaking.
    
    start_time = time.time()
    best_found = best_solution[:]
    
    for _ in range(20):
        if time.time() - start_time > 100:
            break
            
        uncovered = all_mask_np.copy()
        selected = []
        used_blocks = np.zeros(n_blocks, dtype=bool)
        
        while get_uncovered_count(uncovered) > 0:
            gains = get_gain(masks_np, uncovered)
            gains[used_blocks] = -1
            
            best_gain = np.max(gains)
            if best_gain <= 0:
                break
            
            # Find all indices with max gain (or close to max)
            # To introduce randomness, pick from top 5%
            threshold = best_gain * 0.95
            candidates = np.where(gains >= threshold)[0]
            
            if len(candidates) == 0:
                idx = np.argmax(gains)
            else:
                idx = candidates[np.random.randint(len(candidates))]
                
            selected.append(int(idx))
            used_blocks[idx] = True
            uncovered &= ~masks_np[idx]
            
        selected = try_remove_redundant(selected)
        
        if len(selected) < len(best_found):
            best_found = selected[:]
            
    return [list(k_subsets[i]) for i in best_found]
