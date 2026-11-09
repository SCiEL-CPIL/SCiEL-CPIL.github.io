import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    mask_to_idx = {}
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        mask_to_idx[mask] = len(all_masks)
        all_masks.append(mask)
    
    total = len(all_masks)  # 1820
    
    # Precompute pair indices for each mask
    # All pairs in [16]
    all_pairs = []
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pm = (1 << i) | (1 << j)
            pair_to_idx[pm] = idx
            all_pairs.append(pm)
            idx += 1
    num_pairs = len(all_pairs)  # 120
    
    # For each mask, store the list of pair indices it contains
    mask_pair_indices = []
    for m in all_masks:
        indices = []
        for i in range(n):
            if m & (1 << i):
                for j in range(i+1, n):
                    if m & (1 << j):
                        pm = (1 << i) | (1 << j)
                        indices.append(pair_to_idx[pm])
        mask_pair_indices.append(indices)
    
    random.seed(42)
    
    best_family = []
    best_size = 0
    
    for trial in range(100):
        if time.time() > deadline - 8.0:
            break
        
        order = list(range(total))
        random.shuffle(order)
        
        chosen = []
        pair_count = [0] * num_pairs
        
        for idx in order:
            pidxs = mask_pair_indices[idx]
            
            compatible = True
            for pi in pidxs:
                if pair_count[pi] >= 2:
                    compatible = False
                    break
            
            if compatible:
                chosen.append(idx)
                for pi in pidxs:
                    pair_count[pi] += 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = list(chosen)
        
        if time.time() > deadline - 15.0:
            break
    
    # Local search: try to add more sets
    if time.time() < deadline - 10.0:
        chosen_set = set(best_family)
        pair_count = [0] * num_pairs
        for idx in best_family:
            for pi in mask_pair_indices[idx]:
                pair_count[pi] += 1
        
        # Try to add each unchosen set repeatedly until no more can be added
        improved = True
        while improved and time.time() < deadline - 8.0:
            improved = False
            for i in range(total):
                if time.time() > deadline - 8.0:
                    break
                if i in chosen_set:
                    continue
                pidxs = mask_pair_indices[i]
                compatible = True
                for pi in pidxs:
                    if pair_count[pi] >= 2:
                        compatible = False
                        break
                if compatible:
                    best_family.append(i)
                    chosen_set.add(i)
                    for pi in pidxs:
                        pair_count[pi] += 1
                    improved = True
    
    # Convert back to sorted lists
    result = []
    for idx in best_family:
        m = all_masks[idx]
        elems = [i+1 for i in range(n) if m & (1 << i)]
        result.append(sorted(elems))
    
    return result

