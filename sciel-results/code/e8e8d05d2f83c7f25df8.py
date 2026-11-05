import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    full_mask = (1 << n_t) - 1
    
    # 2. Generate all k-blocks and their masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    block_masks = [0] * n_k
    block_tuples = [None] * n_k
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        block_tuples[i] = block
        
    best_solution = None
    best_count = float('inf')
    
    # Try a few random restarts of the greedy algorithm
    # 5 restarts might be enough to get close to optimal
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        covered_mask = 0
        selected_indices = []
        used = [False] * n_k
        
        while covered_mask != full_mask:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_idx = -1
            
            # Optimization: Skip blocks that don't add anything?
            # We must check all to find max gain.
            for i in range(n_k):
                if used[i]:
                    continue
                new_covered = block_masks[i] & (full_mask ^ covered_mask)
                gain = new_covered.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    # Random tie-breaking to explore different paths
                    if random.random() < 0.5:
                        best_idx = i
                        
            if best_idx == -1:
                break # No progress possible?
                
            used[best_idx] = True
            selected_indices.append(best_idx)
            covered_mask |= block_masks[best_idx]
            
        if covered_mask != full_mask:
            continue
            
        # Pruning Phase
        current_indices = selected_indices[:]
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(current_indices)):
                mask_i = block_masks[current_indices[i]]
                others_mask = 0
                for j in range(len(current_indices)):
                    if j != i:
                        others_mask |= block_masks[current_indices[j]]
                        
                if mask_i & ~others_mask == 0:
                    current_indices.pop(i)
                    changed = True
                    break
                    
        if len(current_indices) < best_count:
            best_count = len(current_indices)
            best_solution = [list(block_tuples[idx]) for idx in current_indices]
            
    if best_solution is None:
        # Fallback: Deterministic greedy
        covered_mask = 0
        selected_indices = []
        used = [False] * n_k
        while covered_mask != full_mask:
            best_gain = -1
            best_idx = -1
            for i in range(n_k):
                if used[i]: continue
                gain = (block_masks[i] & (full_mask ^ covered_mask)).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1: break
            used[best_idx] = True
            selected_indices.append(best_idx)
            covered_mask |= block_masks[best_idx]
            
        if covered_mask == full_mask:
             best_solution = [list(block_tuples[idx]) for idx in selected_indices]
             
    return best_solution
