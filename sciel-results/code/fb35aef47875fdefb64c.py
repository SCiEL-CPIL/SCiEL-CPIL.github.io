import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
    
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # 2. Greedy Construction with randomized tie-breaking
    uncovered = np.ones(num_tsubs, dtype=bool)
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        max_gain = gains.max()
        if max_gain == 0:
            break
        
        # Randomized tie-breaking among blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        b = candidates[random.randint(0, len(candidates) - 1)]
        
        selected_block_indices.append(b)
        
        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                gains[tsub_to_blocks_np[s_idx]] -= 1
                
    if not uncovered.any():
        # 3. Local Optimization
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        def is_redundant(b_idx):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        def find_replacement(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None
            
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in unique_tsubs:
                    if s not in c_set:
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)
        
        def find_2for1_replacement(b_out1, b_out2, exclude_set):
            """Find a single block that covers all unique t-subsets of b_out1 and b_out2."""
            s_uniq1 = [s for s in block_to_tsubs[b_out1] if cover_count[s] == 1]
            s_uniq2 = [s for s in block_to_tsubs[b_out2] if cover_count[s] == 1]
            
            # Union of unique t-subsets
            all_unique = list(set(s_uniq1 + s_uniq2))
            if not all_unique:
                return None
            
            first_s = all_unique[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in all_unique:
                    if s not in c_set:
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)
        
        improved = True
        no_improve_count = 0
        max_no_improve = 50
        
        while improved and time.time() < deadline:
            improved = False
            
            # Step A: Redundancy Removal
            redundant_blocks = []
            for b in selected_block_indices:
                if is_redundant(b):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                no_improve_count = 0
                continue
            
            # Compute unique t-subsets for each block
            unique_tsubs_map = {}
            unique_counts = {}
            for b in selected_block_indices:
                u_list = [s for s in block_to_tsubs[b] if cover_count[s] == 1]
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            # Step B: 1-for-1 Swap
            min_unique = min(unique_counts.values()) if unique_counts else 0
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs_map[b_out]
                if not s_uniq:
                    continue
                exclude_set = set(selected_block_indices)
                b_in = find_replacement(s_uniq, exclude_set)
                if b_in is not None:
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    swap_made = True
                    improved = True
                    no_improve_count = 0
                    break
            
            if swap_made:
                continue
            
            # Step C: 2-for-1 Swap (remove two blocks, add one)
            if len(selected_block_indices) >= 2:
                # Find pairs of blocks with low combined unique count
                sorted_blocks = sorted(selected_block_indices, key=lambda b: unique_counts[b])
                pair_found = False
                
                for i in range(len(sorted_blocks)):
                    if time.time() > deadline - 5:
                        break
                    b1 = sorted_blocks[i]
                    if unique_counts[b1] > 5:
                        break
                    for j in range(i + 1, len(sorted_blocks)):
                        b2 = sorted_blocks[j]
                        if unique_counts[b1] + unique_counts[b2] > 10:
                            break
                        
                        exclude_set = set(selected_block_indices)
                        b_in = find_2for1_replacement(b1, b2, exclude_set)
                        
                        if b_in is not None:
                            selected_block_indices.remove(b1)
                            selected_block_indices.remove(b2)
                            selected_block_indices.append(b_in)
                            for s in block_to_tsubs[b1]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b2]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b_in]:
                                cover_count[s] += 1
                            pair_found = True
                            improved = True
                            no_improve_count = 0
                            break
                    if pair_found:
                        break
                
                if pair_found:
                    continue
            
            # No improvement this iteration
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        
        result = [list(all_k_subsets[i]) for i in selected_block_indices]
        return result
    else:
        return []
