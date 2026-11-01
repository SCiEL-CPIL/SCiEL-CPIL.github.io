import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
    
    # Inverse: which blocks cover each t-subset
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Precompute block masks for fast coverage updates
    block_masks = []
    for tsubs in block_to_tsubs:
        mask = np.zeros(num_tsubs, dtype=bool)
        for s in tsubs:
            mask[s] = True
        block_masks.append(mask)
    
    def greedy_construction(randomize_ties=False, random_seed=None):
        """
        Greedy set cover with incremental gains update.
        randomize_ties: if True, randomly pick from top-k gainers to break symmetry.
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(num_tsubs, dtype=bool)
        k_choose_t = len(block_to_tsubs[0])
        gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
        
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            # Find block with max gain
            if randomize_ties and random.random() < 0.25:
                # Pick randomly from top 20 gainers to break symmetry
                top_k = 20
                if num_blocks > top_k:
                    top_indices = np.argpartition(gains, -top_k)[-top_k:]
                    # Filter to only those with positive gain
                    positive_top = [idx for idx in top_indices if gains[idx] > 0]
                    if positive_top:
                        b = random.choice(positive_top)
                    else:
                        b = int(np.argmax(gains))
                else:
                    b = int(np.argmax(gains))
            else:
                b = int(np.argmax(gains))
            
            if gains[b] == 0:
                return None
            
            selected_block_indices.append(b)
            
            # Update uncovered and gains incrementally
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    # Decrement gains for all blocks that cover this t-subset
                    gains[tsub_to_blocks_np[s_idx]] -= 1
        
        if uncovered.any():
            return None
        
        return selected_block_indices
    
    def local_optimize(blocks, max_iterations=50):
        """
        Local search with redundancy removal, 1-swap, and 2-swap moves.
        """
        if not blocks:
            return []
        
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        # Build cover_count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        def is_redundant(b_idx):
            """Check if removing b_idx still leaves all tsubs covered"""
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        def find_replacement_for_unique(unique_tsubs, exclude_set, max_candidates=30):
            """Find a block that covers all unique_tsubs and is not in exclude_set"""
            if not unique_tsubs:
                return None
            
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid = []
            for c in candidates:
                if c in exclude_set:
                    continue
                c_set = block_to_tsubs_set[c]
                all_covered = True
                for s in unique_tsubs:
                    if s not in c_set:
                        all_covered = False
                        break
                if all_covered:
                    valid.append(c)
                    if len(valid) >= max_candidates:
                        break
            
            if not valid:
                return None
            return random.choice(valid)
        
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
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step 2: 1-swap - try to replace one block with a better one
            unique_counts = {}
            unique_tsubs_map = {}
            for b in selected_list:
                u_list = [s for s in block_to_tsubs[b] if cover_count[s] == 1]
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            if selected_list:
                min_unique = min(unique_counts.values())
                candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
                random.shuffle(candidates_out)
                
                swap_made = False
                for b_out in candidates_out:
                    s_uniq = unique_tsubs_map[b_out]
                    b_in = find_replacement_for_unique(s_uniq, selected_set)
                    if b_in is not None:
                        selected_list.remove(b_out)
                        selected_set.remove(b_out)
                        selected_list.append(b_in)
                        selected_set.add(b_in)
                        for s in block_to_tsubs[b_out]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b_in]:
                            cover_count[s] += 1
                        swap_made = True
                        improved = True
                        break
                
                if swap_made:
                    continue
            
            # Step 3: 2-swap - try to replace 2 blocks with 1 or 2 blocks
            if len(selected_list) >= 2 and time.time() < deadline:
                n_pairs = min(30, len(selected_list) * (len(selected_list) - 1) // 2)
                pairs = list(itertools.combinations(range(len(selected_list)), 2))
                random.shuffle(pairs)
                
                two_swap_made = False
                for i, j in pairs[:n_pairs]:
                    if time.time() > deadline:
                        break
                    
                    b1 = selected_list[i]
                    b2 = selected_list[j]
                    
                    # Find unique tsubs that would be uncovered if we remove both
                    temp_cover = cover_count.copy()
                    for s in block_to_tsubs[b1]:
                        temp_cover[s] -= 1
                    for s in block_to_tsubs[b2]:
                        temp_cover[s] -= 1
                    
                    unique_after_removal = [s for s in range(num_tsubs) if temp_cover[s] == 0]
                    
                    if not unique_after_removal:
                        # Both blocks are redundant together
                        selected_list.remove(b1)
                        selected_list.remove(b2)
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        for s in block_to_tsubs[b1]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b2]:
                            cover_count[s] -= 1
                        two_swap_made = True
                        improved = True
                        break
                    
                    # Try to find 1 block that covers all unique_after_removal
                    b_in1 = find_replacement_for_unique(unique_after_removal, selected_set, max_candidates=20)
                    if b_in1 is not None:
                        selected_list.remove(b1)
                        selected_list.remove(b2)
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        selected_list.append(b_in1)
                        selected_set.add(b_in1)
                        for s in block_to_tsubs[b1]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b2]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b_in1]:
                            cover_count[s] += 1
                        two_swap_made = True
                        improved = True
                        break
                
                if two_swap_made:
                    continue
            
            # No improvement found
            break
        
        return selected_list
    
    best_solution = None
    best_len = float('inf')
    
    # Multi-start with different randomizations
    num_starts = 12
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        random.seed(i * 999983 + 7)  # Different seed for diversity
        randomize = (i % 3 != 0)  # Mix deterministic and randomized greedy
        
        blocks = greedy_construction(randomize_ties=randomize, random_seed=i * 31337 + 11)
        
        if blocks is None:
            continue
        
        optimized_blocks = local_optimize(blocks)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
        
        if best_len <= 150:
            break
    
    if best_solution is None:
        blocks = greedy_construction(randomize_ties=False)
        if blocks:
            best_solution = local_optimize(blocks)
        else:
            return []
    
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
