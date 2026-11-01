import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage
    block_to_tsubs = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
    
    # Inverse: which blocks cover each t-subset
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Use numpy for efficient bit operations
    # Represent coverage as numpy boolean arrays
    block_masks = []
    for tsubs in block_to_tsubs:
        mask = np.zeros(num_tsubs, dtype=bool)
        for s in tsubs:
            mask[s] = True
        block_masks.append(mask)
    
    # Precompute block_masks as a 2D numpy array for faster operations
    block_masks_2d = np.array(block_masks, dtype=np.int8)
    
    def greedy_construction(strategy='max_gain', weight=None):
        """
        strategy: 'max_gain' - pick block with max new coverage
                  'min_uncovered' - pick block covering most currently uncovered
                  'weighted' - use custom weights for t-subsets
        """
        uncovered = np.ones(num_tsubs, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            if strategy == 'max_gain':
                # For each block, count how many currently uncovered tsubs it covers
                gains = block_masks_2d @ uncovered.astype(np.int8)
                best_block = int(np.argmax(gains))
                if gains[best_block] == 0:
                    return None
            elif strategy == 'min_uncovered':
                # Pick a random uncovered tsub, then pick best block covering it
                uncovered_indices = np.where(uncovered)[0]
                s_idx = int(uncovered_indices[random.randint(0, len(uncovered_indices)-1)])
                candidates = tsub_to_blocks_np[s_idx]
                gains = block_masks_2d[candidates] @ uncovered.astype(np.int8)
                best_in_candidates = int(candidates[np.argmax(gains)])
                if gains[np.argmax(gains)] == 0:
                    return None
                best_block = best_in_candidates
            elif strategy == 'weighted':
                # Weighted greedy: prioritize t-subsets that are hard to cover
                if weight is None:
                    weight = np.ones(num_tsubs, dtype=np.float64)
                # Update weights: uncovered tsubs with fewer covering blocks get higher weight
                if not hasattr(greedy_construction, '_weight_cache'):
                    greedy_construction._weight_cache = {}
                
                gains = (block_masks_2d @ (uncovered.astype(np.float64) * weight)).astype(np.int64)
                best_block = int(np.argmax(gains))
                if gains[best_block] == 0:
                    return None
            else:
                # Default to max_gain
                gains = block_masks_2d @ uncovered.astype(np.int8)
                best_block = int(np.argmax(gains))
                if gains[best_block] == 0:
                    return None
            
            selected.append(best_block)
            uncovered[block_masks[best_block]] = False
        
        return selected
    
    def local_optimize(blocks, max_iterations=50):
        """
        Aggressive local search with 1-swap, 2-swap, and 3-swap moves.
        """
        if not blocks:
            return []
        
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        # Build cover_count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            cover_count[block_to_tsubs[b]] += 1
        
        def is_redundant(b_idx):
            """Check if removing b_idx still leaves all tsubs covered"""
            tsubs_b = block_to_tsubs[b_idx]
            return np.all(cover_count[tsubs_b] > 1)
        
        def find_replacement_for_unique(unique_tsubs, exclude_set, max_candidates=100):
            """Find a block that covers all unique_tsubs and is not in exclude_set"""
            if not unique_tsubs:
                return None
            
            # Use the first unique tsub to get candidates
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid = []
            for c in candidates:
                if c in exclude_set:
                    continue
                # Check if c covers all unique tsubs
                c_mask = block_masks[c]
                all_covered = np.all(c_mask[unique_tsubs])
                if all_covered:
                    valid.append(c)
                    if len(valid) >= max_candidates:
                        break
            
            if not valid:
                return None
            return random.choice(valid)
        
        def try_2swap_replacement(b1, b2, cover_count_current, selected_set_current):
            """Try to find 1 or 2 blocks that replace b1 and b2"""
            # Find unique tsubs that would be uncovered if we remove both
            temp_cover = cover_count_current.copy()
            temp_cover[block_to_tsubs[b1]] -= 1
            temp_cover[block_to_tsubs[b2]] -= 1
            
            unique_after_removal = np.where(temp_cover == 0)[0]
            
            if len(unique_after_removal) == 0:
                # Both blocks are redundant together - remove them
                return ('remove_both', None, None)
            
            # Try to find 1 block that covers all unique_after_removal
            b_in1 = find_replacement_for_unique(unique_after_removal, selected_set_current, max_candidates=50)
            if b_in1 is not None:
                return ('replace_with_1', b_in1, None)
            
            # Try to find 2 blocks that cover all unique_after_removal
            # Split unique_after_removal into two groups and find blocks for each
            if len(unique_after_removal) <= 14:  # Limit for efficiency
                # Try different splits
                n = len(unique_after_removal)
                splits_tried = 0
                max_splits = 20
                
                while splits_tried < max_splits and time.time() < deadline:
                    # Random split
                    if n <= 2:
                        break
                    split_point = random.randint(1, n - 1)
                    group1 = unique_after_removal[:split_point]
                    group2 = unique_after_removal[split_point:]
                    
                    b_in2a = find_replacement_for_unique(group1, selected_set_current, max_candidates=30)
                    if b_in2a is not None:
                        b_in2b = find_replacement_for_unique(group2, selected_set_current, max_candidates=30)
                        if b_in2b is not None and b_in2b != b_in2a:
                            return ('replace_with_2', b_in2a, b_in2b)
                    splits_tried += 1
            
            return None
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            
            # Step 1: Remove redundant blocks
            to_remove = [b for b in selected_list if is_redundant(b)]
            if to_remove:
                for b in to_remove:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    cover_count[block_to_tsubs[b]] -= 1
                improved = True
                continue
            
            # Step 2: 1-swap - try to replace one block with a better one
            # Find block with minimum unique coverage
            unique_counts = {}
            unique_tsubs_map = {}
            for b in selected_list:
                tsubs_b = block_to_tsubs[b]
                u_mask = cover_count[tsubs_b] == 1
                u_list = tsubs_b[u_mask]
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            if selected_list:
                min_unique = min(unique_counts.values())
                candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
                random.shuffle(candidates_out)
                
                swap_made = False
                for b_out in candidates_out:
                    if time.time() > deadline:
                        break
                    s_uniq = unique_tsubs_map[b_out]
                    b_in = find_replacement_for_unique(s_uniq, selected_set)
                    if b_in is not None:
                        # Perform swap
                        selected_list.remove(b_out)
                        selected_set.remove(b_out)
                        selected_list.append(b_in)
                        selected_set.add(b_in)
                        cover_count[block_to_tsubs[b_out]] -= 1
                        cover_count[block_to_tsubs[b_in]] += 1
                        swap_made = True
                        improved = True
                        break
                
                if swap_made:
                    continue
            
            # Step 3: 2-swap - try to replace 2 blocks with 1 or 2 blocks
            if len(selected_list) >= 2 and time.time() < deadline:
                # Try removing pairs and finding replacements
                n_pairs = min(40, len(selected_list) * (len(selected_list) - 1) // 2)
                pairs = list(itertools.combinations(range(len(selected_list)), 2))
                random.shuffle(pairs)
                
                two_swap_made = False
                for i, j in pairs[:n_pairs]:
                    if time.time() > deadline:
                        break
                    
                    b1 = selected_list[i]
                    b2 = selected_list[j]
                    
                    result = try_2swap_replacement(b1, b2, cover_count, selected_set)
                    
                    if result is not None:
                        if result[0] == 'remove_both':
                            selected_list.remove(b1)
                            selected_list.remove(b2)
                            selected_set.remove(b1)
                            selected_set.remove(b2)
                            cover_count[block_to_tsubs[b1]] -= 1
                            cover_count[block_to_tsubs[b2]] -= 1
                        elif result[0] == 'replace_with_1':
                            b_in1 = result[1]
                            selected_list.remove(b1)
                            selected_list.remove(b2)
                            selected_set.remove(b1)
                            selected_set.remove(b2)
                            selected_list.append(b_in1)
                            selected_set.add(b_in1)
                            cover_count[block_to_tsubs[b1]] -= 1
                            cover_count[block_to_tsubs[b2]] -= 1
                            cover_count[block_to_tsubs[b_in1]] += 1
                        elif result[0] == 'replace_with_2':
                            b_in1 = result[1]
                            b_in2 = result[2]
                            selected_list.remove(b1)
                            selected_list.remove(b2)
                            selected_set.remove(b1)
                            selected_set.remove(b2)
                            selected_list.append(b_in1)
                            selected_list.append(b_in2)
                            selected_set.add(b_in1)
                            selected_set.add(b_in2)
                            cover_count[block_to_tsubs[b1]] -= 1
                            cover_count[block_to_tsubs[b2]] -= 1
                            cover_count[block_to_tsubs[b_in1]] += 1
                            cover_count[block_to_tsubs[b_in2]] += 1
                        
                        two_swap_made = True
                        improved = True
                        break
                
                if two_swap_made:
                    continue
            
            # Step 4: 3-swap - try to replace 3 blocks with 1 or 2 blocks
            if len(selected_list) >= 3 and time.time() < deadline:
                n_triples = min(20, len(selected_list) * (len(selected_list) - 1) * (len(selected_list) - 2) // 6)
                triples = list(itertools.combinations(range(len(selected_list)), 3))
                random.shuffle(triples)
                
                three_swap_made = False
                for i, j, l in triples[:n_triples]:
                    if time.time() > deadline:
                        break
                    
                    b1 = selected_list[i]
                    b2 = selected_list[j]
                    b3 = selected_list[l]
                    
                    # Find unique tsubs that would be uncovered if we remove all three
                    temp_cover = cover_count.copy()
                    temp_cover[block_to_tsubs[b1]] -= 1
                    temp_cover[block_to_tsubs[b2]] -= 1
                    temp_cover[block_to_tsubs[b3]] -= 1
                    
                    unique_after_removal = np.where(temp_cover == 0)[0]
                    
                    if len(unique_after_removal) == 0:
                        # All three blocks are redundant together - remove them
                        selected_list.remove(b1)
                        selected_list.remove(b2)
                        selected_list.remove(b3)
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        selected_set.remove(b3)
                        cover_count[block_to_tsubs[b1]] -= 1
                        cover_count[block_to_tsubs[b2]] -= 1
                        cover_count[block_to_tsubs[b3]] -= 1
                        three_swap_made = True
                        improved = True
                        break
                    
                    # Try to find 1 or 2 blocks that cover all unique_after_removal
                    b_in1 = find_replacement_for_unique(unique_after_removal, selected_set, max_candidates=30)
                    if b_in1 is not None:
                        selected_list.remove(b1)
                        selected_list.remove(b2)
                        selected_list.remove(b3)
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        selected_set.remove(b3)
                        selected_list.append(b_in1)
                        selected_set.add(b_in1)
                        cover_count[block_to_tsubs[b1]] -= 1
                        cover_count[block_to_tsubs[b2]] -= 1
                        cover_count[block_to_tsubs[b3]] -= 1
                        cover_count[block_to_tsubs[b_in1]] += 1
                        three_swap_made = True
                        improved = True
                        break
                    else:
                        # Try 2 blocks
                        if len(unique_after_removal) <= 14:
                            n = len(unique_after_removal)
                            splits_tried = 0
                            max_splits = 15
                            
                            while splits_tried < max_splits and time.time() < deadline:
                                if n <= 2:
                                    break
                                split_point = random.randint(1, n - 1)
                                group1 = unique_after_removal[:split_point]
                                group2 = unique_after_removal[split_point:]
                                
                                b_in2a = find_replacement_for_unique(group1, selected_set, max_candidates=20)
                                if b_in2a is not None:
                                    b_in2b = find_replacement_for_unique(group2, selected_set, max_candidates=20)
                                    if b_in2b is not None and b_in2b != b_in2a:
                                        selected_list.remove(b1)
                                        selected_list.remove(b2)
                                        selected_list.remove(b3)
                                        selected_set.remove(b1)
                                        selected_set.remove(b2)
                                        selected_set.remove(b3)
                                        selected_list.append(b_in2a)
                                        selected_list.append(b_in2b)
                                        selected_set.add(b_in2a)
                                        selected_set.add(b_in2b)
                                        cover_count[block_to_tsubs[b1]] -= 1
                                        cover_count[block_to_tsubs[b2]] -= 1
                                        cover_count[block_to_tsubs[b3]] -= 1
                                        cover_count[block_to_tsubs[b_in2a]] += 1
                                        cover_count[block_to_tsubs[b_in2b]] += 1
                                        three_swap_made = True
                                        improved = True
                                        break
                                splits_tried += 1
                            
                            if three_swap_made:
                                break
                
                if three_swap_made:
                    continue
            
            # No improvement found
            break
        
        return selected_list
    
    def verify_solution(blocks):
        """Verify that the solution covers all t-subsets"""
        covered = np.zeros(num_tsubs, dtype=bool)
        for b in blocks:
            covered[block_to_tsubs[b]] = True
        return np.all(covered)
    
    best_solution = None
    best_len = float('inf')
    
    # Multi-start with different strategies and seeds
    strategies = ['max_gain', 'min_uncovered', 'max_gain', 'min_uncovered', 'weighted', 'max_gain', 'min_uncovered', 'weighted']
    num_starts = 12
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        strategy = strategies[i % len(strategies)]
        random.seed(i * 12345 + 678)  # Different seed for diversity
        
        blocks = greedy_construction(strategy=strategy)
        
        if blocks is None:
            continue
        
        optimized_blocks = local_optimize(blocks, max_iterations=40)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
        
        if best_len <= 151:
            break
    
    if best_solution is None:
        blocks = greedy_construction(strategy='max_gain')
        if blocks:
            best_solution = local_optimize(blocks, max_iterations=30)
        else:
            return []
    
    # Final polish: one more round of optimization on the best solution
    if time.time() < deadline - 5:
        polished = local_optimize(best_solution, max_iterations=20)
        if len(polished) < best_len:
            best_solution = polished
            best_len = len(polished)
    
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
