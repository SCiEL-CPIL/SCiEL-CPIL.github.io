import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # Precompute frequency of each t-subset across all k-subsets (for weighted greedy)
    tsub_freq = [0] * n_tsubs
    for j in range(n_ksubs):
        mask = ksub_masks[j]
        # Count bits per position - use a different approach
        # For each t-subset, count how many k-subsets cover it
        for tsub in itertools.combinations(range(v), t):
            idx = tsub_to_idx[tsub]
            if mask & (1 << idx):
                tsub_freq[idx] += 1

    # Helper: Weighted Greedy Set Cover
    def weighted_greedy_init(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered_mask = (1 << n_tsubs) - 1
        remaining = n_tsubs
        blocks = []
        used_blocks = set()
        
        while remaining > 0:
            if time.time() > deadline:
                break

            best_j = -1
            best_score = -1
            candidates = []

            for j in range(n_ksubs):
                if j in used_blocks:
                    continue
                
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                if gain == 0:
                    continue
                
                # Weighted score: prioritize covering rare t-subsets
                # Use harmonic mean or sum of 1/freq for uncovered t-subsets in this block
                # For efficiency, approximate with gain * (1 / avg_freq_of_covered)
                # Or simpler: sum of weights where weight[i] = 1/tsub_freq[i]
                # Precompute weights
                score = 0
                cm = common
                while cm:
                    # Get lowest set bit
                    lsb = cm & (-cm)
                    idx = lsb.bit_length() - 1
                    if tsub_freq[idx] > 0:
                        score += 1.0 / tsub_freq[idx]
                    cm ^= lsb
                
                if score > best_score + 1e-10:
                    best_score = score
                    best_j = j
                    candidates = [j]
                elif abs(score - best_score) < 1e-10:
                    candidates.append(j)

            if best_j == -1:
                break

            if randomize and len(candidates) > 1:
                best_j = random.choice(candidates)

            blocks.append(list(all_ksubs[best_j]))
            used_blocks.add(best_j)
            
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= (uncovered_mask.bit_count() if False else 0)  # Recalculate
            remaining = n_tsubs - uncovered_mask.bit_count()

        return blocks, used_blocks

    # Helper: Local Search / Polish with improved swap search
    def local_search(initial_blocks, initial_indices, time_budget=None):
        if time_budget is None:
            time_budget = deadline - 2.0
        
        if time.time() > time_budget:
            return initial_blocks, initial_indices

        active_indices = set(initial_indices)
        
        # Fast unique mask computation using counting
        def get_unique_masks_fast(active_set):
            """Compute unique coverage for each block using bit counting"""
            unique_masks = {}
            active_list = list(active_set)
            n_active = len(active_list)
            
            # For each t-subset, count how many active blocks cover it
            # Then a t-subset is unique to block b if count == 1 and b covers it
            
            # Build coverage count array
            cover_count = [0] * n_tsubs
            # This is O(n_active * n_tsubs_in_block) which might be slow
            # Alternative: use mask operations
            
            # For each block, compute its unique mask
            for idx in active_set:
                my_mask = ksub_masks[idx]
                # Count how many other blocks also cover each bit in my_mask
                # For each bit position in my_mask, check if any other block has it
                unique = 0
                cm = my_mask
                while cm:
                    lsb = cm & (-cm)
                    bit_pos = lsb.bit_length() - 1
                    # Check if any other active block covers this bit
                    covered_by_other = False
                    for idx2 in active_set:
                        if idx2 != idx and (ksub_masks[idx2] & lsb):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        unique |= lsb
                    cm ^= lsb
                unique_masks[idx] = unique
            
            return unique_masks

        improved = True
        iteration = 0
        while improved and time.time() < time_budget:
            improved = False
            iteration += 1
            
            # Calculate unique masks for all active blocks
            unique_masks = get_unique_masks_fast(active_indices)
            
            # Find redundant blocks (unique mask is 0)
            to_remove = [idx for idx, um in unique_masks.items() if um == 0]
            if to_remove:
                for idx in to_remove:
                    active_indices.remove(idx)
                improved = True
                continue
            
            # Try to find a swap that enables a removal
            # Sort by unique coverage size to prioritize "weak" blocks
            sorted_indices = sorted(active_indices, key=lambda x: unique_masks[x].bit_count())
            
            for target_idx in sorted_indices:
                if time.time() > time_budget:
                    break
                
                target_unique = unique_masks[target_idx]
                if target_unique == 0:
                    continue
                
                # Find a replacement block that covers target_unique and has good overlap
                best_replacement = -1
                best_overlap = -1
                target_mask = ksub_masks[target_idx]
                
                # Iterate through all blocks not in active set
                for j in range(n_ksubs):
                    if j in active_indices:
                        continue
                    if (ksub_masks[j] & target_unique) == target_unique:
                        # This block can replace target_idx
                        # Prefer blocks with high overlap with target (to maximize redundancy)
                        overlap = (ksub_masks[j] & target_mask).bit_count()
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_replacement = j
                
                if best_replacement != -1:
                    # Perform swap
                    active_indices.remove(target_idx)
                    active_indices.add(best_replacement)
                    
                    # Check if this swap allows any removal
                    new_unique_masks = get_unique_masks_fast(active_indices)
                    
                    new_redundants = [idx for idx, um in new_unique_masks.items() if um == 0]
                    if new_redundants:
                        for idx in new_redundants:
                            active_indices.remove(idx)
                        improved = True
                        break  # Restart the main loop with the new set
                    else:
                        # Revert if no immediate improvement
                        active_indices.remove(best_replacement)
                        active_indices.add(target_idx)
                
                # If no swap found for this target, try add-minimize
                # Try adding a block that creates redundancy, then remove
                # This is more complex, skip for now

        if active_indices:
            final_blocks = [list(all_ksubs[idx]) for idx in active_indices]
            final_mask = 0
            for idx in active_indices:
                final_mask |= ksub_masks[idx]
            if final_mask == (1 << n_tsubs) - 1:
                return final_blocks, list(active_indices)
        
        return initial_blocks, initial_indices

    # 4. Multi-start strategy
    best_blocks = None
    best_count = float('inf')
    
    # First run: Weighted Greedy
    blocks, indices = weighted_greedy_init(randomize=False)
    polished_blocks, polished_indices = local_search(blocks, indices)
    if polished_blocks and len(polished_blocks) < best_count:
        best_blocks = polished_blocks
        best_count = len(polished_blocks)
        
    # Subsequent runs: Randomized Weighted Greedy
    # Limit number of starts based on time
    max_starts = 15
    for i in range(max_starts):
        if time.time() > deadline - 15.0:
            break
            
        # Randomized weighted greedy with different seeds
        rand_blocks, rand_indices = weighted_greedy_init(randomize=True, seed=i * 12345 + 67890)
        
        # Polish with time budget
        remaining_time = deadline - 5.0 - time.time()
        if remaining_time < 5.0:
            break
        polished_rand_blocks, _ = local_search(rand_blocks, rand_indices, time_budget=time.time() + remaining_time)
        
        if polished_rand_blocks and len(polished_rand_blocks) < best_count:
            best_blocks = polished_rand_blocks
            best_count = len(polished_rand_blocks)

    if best_blocks is None:
        # Fallback to first result if multi-start failed or timed out
        blocks, indices = weighted_greedy_init(randomize=False)
        best_blocks, _ = local_search(blocks, indices)

    return best_blocks
