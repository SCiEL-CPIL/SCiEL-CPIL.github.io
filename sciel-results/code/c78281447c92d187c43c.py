import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(all_points, t))
    n_t_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(all_points, k))
    n_k_subsets = len(k_subsets)
    
    block_masks = [0] * n_k_subsets
    for idx, k_sub in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx_t = subset_to_idx[t_sub]
            mask |= (1 << idx_t)
        block_masks[idx] = mask
        
    target_mask = (1 << n_t_subsets) - 1
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        covered_mask = 0
        selected_indices = []
        remaining_indices = set(range(n_k_subsets))
        
        while covered_mask != target_mask:
            best_idx = -1
            best_gain = -1
            
            for i in remaining_indices:
                gain_mask = block_masks[i] & ~covered_mask
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if best_gain == 0:
                        break
            
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            covered_mask |= block_masks[best_idx]
            remaining_indices.discard(best_idx)
            
        return selected_indices

    def remove_redundant(selected_indices):
        if not selected_indices:
            return []
            
        counts = [0] * n_t_subsets
        
        for idx in selected_indices:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                counts[idx_t] += 1
                m ^= lsb
        
        changed = True
        while changed:
            changed = False
            new_selected = []
            for idx in selected_indices:
                mask = block_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if counts[idx_t] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        counts[idx_t] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_selected.append(idx)
            
            selected_indices = new_selected
            
        return [k_subsets[i] for i in selected_indices]

    best_blocks = None
    best_len = float('inf')
    start_time = time.time()
    time_limit = 115
    
    seed = 42
    while time.time() - start_time < time_limit:
        indices = greedy_construct(seed)
        if not indices:
            break
        blocks = remove_redundant(indices)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        seed += 1
        
    return best_blocks
