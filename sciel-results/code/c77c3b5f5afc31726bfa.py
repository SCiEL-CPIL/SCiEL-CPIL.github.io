import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Pre-compute all t-subsets of v points
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map each subset to an index for vectorization
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    def block_to_mask(block):
        mask = np.zeros(n_subsets, dtype=bool)
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask[idx] = True
        return mask
    
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_subsets, dtype=bool)
        blocks = []
        point_freq = np.zeros(v, dtype=np.int32)
        
        while np.any(uncovered) and time.time() < deadline - 15:
            # Find the first uncovered subset
            idx = np.where(uncovered)[0][0]
            first_uncovered = all_t_subsets[idx]
            block = list(first_uncovered)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                best_freq = float('inf')
                
                in_block = set(block)
                candidates = [x for x in range(v) if x not in in_block]
                
                if not candidates:
                    break
                
                block_pairs = list(itertools.combinations(block, t - 1))
                
                gains = []
                for x in candidates:
                    count = 0
                    for pair in block_pairs:
                        sub = tuple(sorted(pair + (x,)))
                        idx_sub = subset_to_idx[sub]
                        if uncovered[idx_sub]:
                            count += 1
                    if count > 0:
                        gains.append((count, -point_freq[x], x))
                    else:
                        gains.append((0, -point_freq[x], x))
                
                gains.sort(key=lambda x: (x[0], x[1]), reverse=True)
                
                best_point = gains[0][2]
                block.append(best_point)
                point_freq[best_point] += 1
            
            block.sort()
            blocks.append(block)
            
            mask = block_to_mask(block)
            uncovered &= ~mask
            
        return blocks
    
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        n_blocks = len(blocks)
        masks = [block_to_mask(b) for b in blocks]
        
        removed = [False] * n_blocks
        changed = True
        
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(n_blocks):
                if removed[i]:
                    continue
                
                redundant = True
                subs_in_i = np.where(masks[i])[0]
                for s_idx in subs_in_i:
                    covered = False
                    for j in range(n_blocks):
                        if j == i or removed[j]:
                            continue
                        if masks[j][s_idx]:
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                
                if redundant:
                    removed[i] = True
                    changed = True
        
        return [b for i, b in enumerate(blocks) if not removed[i]]
    
    def local_search(blocks, seed=0):
        rng = random.Random(seed)
        current_blocks = blocks[:]
        current_size = len(current_blocks)
        
        masks = [block_to_mask(b) for b in current_blocks]
        coverage_counts = np.sum(masks, axis=0)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            for i in range(len(current_blocks)):
                if time.time() > deadline - 10:
                    break
                
                lost_mask = masks[i]
                coverage_counts -= lost_mask
                newly_uncovered = np.where(coverage_counts == 0)[0]
                
                if len(newly_uncovered) == 0:
                    current_blocks.pop(i)
                    masks.pop(i)
                    current_size -= 1
                    improved = True
                    break
                else:
                    union_set = set()
                    for idx in newly_uncovered:
                        union_set.update(all_t_subsets[idx])
                    
                    if len(union_set) <= k:
                        candidate_block = sorted(list(union_set))
                        while len(candidate_block) < k:
                            candidate_block.append(rng.randint(0, v-1))
                        candidate_block = sorted(list(set(candidate_block)))[:k]
                        
                        new_mask = block_to_mask(candidate_block)
                        
                        if np.all(new_mask[newly_uncovered]):
                            tentative_blocks = current_blocks + [candidate_block]
                            tentative_blocks = remove_redundant(tentative_blocks)
                            
                            if len(tentative_blocks) < current_size:
                                current_blocks = tentative_blocks
                                current_size = len(current_blocks)
                                improved = True
                                break
                    coverage_counts += lost_mask
        
        return current_blocks
    
    # Main loop with restarts
    best_blocks = None
    best_size = float('inf')
    
    # Run greedy multiple times with different random seeds
    for restart in range(30):
        if time.time() > deadline - 20:
            break
        
        greedy_blocks = greedy_cover(restart)
        cleaned_blocks = remove_redundant(greedy_blocks)
        
        if time.time() < deadline - 15:
            local_blocks = local_search(cleaned_blocks, restart + 1000)
        else:
            local_blocks = cleaned_blocks
            
        if len(local_blocks) < best_size:
            best_size = len(local_blocks)
            best_blocks = local_blocks[:]
    
    if best_blocks is None:
        best_blocks = remove_redundant(greedy_cover(0))
    
    return best_blocks
