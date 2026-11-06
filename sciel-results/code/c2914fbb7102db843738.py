import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Precompute all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n_sets = len(all_sets)  # 1820
    
    # Precompute pairwise intersection masks
    # inter[i][j] = all_sets[i] & all_sets[j]
    # This is 1820*1820 = ~3.3M entries, each an int - should be fine
    
    def popcount(x):
        return bin(x).count('1')
    
    # Check if three sets form a sunflower: A∩B == A∩C == B∩C
    def is_sunflower(a, b, c):
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def greedy_randomized(seed):
        rng = random.Random(seed)
        chosen = []
        chosen_masks = []
        # Shuffle
        indices = list(range(n_sets))
        rng.shuffle(indices)
        
        for idx in indices:
            mask = all_sets[idx]
            ok = True
            for j in range(len(chosen_masks)):
                for k in range(j+1, len(chosen_masks)):
                    if is_sunflower(mask, chosen_masks[j], chosen_masks[k]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(idx)
                chosen_masks.append(mask)
        
        return chosen
    
    # Local search: try to add a set, or swap
    def try_add(chosen, all_indices):
        chosen_set = set(chosen)
        chosen_masks = [all_sets[i] for i in chosen]
        for idx in all_indices:
            if idx in chosen_set:
                continue
            mask = all_sets[idx]
            ok = True
            for j in range(len(chosen_masks)):
                for k in range(j+1, len(chosen_masks)):
                    if is_sunflower(mask, chosen_masks[j], chosen_masks[k]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return idx
        return None
    
    best_chosen = []
    best_size = 0
    
    # Try multiple random greedy restarts
    for seed in range(200):
        if time.time() > deadline:
            break
        chosen = greedy_randomized(seed)
        if len(chosen) > best_size:
            best_chosen = chosen
            best_size = len(chosen)
    
    # Local search on best
    for iteration in range(50):
        if time.time() > deadline:
            break
        chosen_set = set(best_chosen)
        # Try to add
        added = try_add(best_chosen, list(range(n_sets)))
        if added is not None:
            best_chosen.append(added)
            best_size += 1
            continue
        # Try swaps: remove one, add two
        improved = False
        for i in range(len(best_chosen)):
            if time.time() > deadline:
                break
            removed = best_chosen[i]
            temp_chosen = [x for j, x in enumerate(best_chosen) if j != i]
            temp_set = set(temp_chosen)
            added_count = 0
            added_indices = []
            for idx in range(n_sets):
                if idx in temp_set or idx == removed:
                    continue
                mask = all_sets[idx]
                ok = True
                temp_masks = [all_sets[x] for x in temp_chosen]
                for j in range(len(temp_masks)):
                    for k in range(j+1, len(temp_masks)):
                        if is_sunflower(mask, temp_masks[j], temp_masks[k]):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    added_indices.append(idx)
                    temp_set.add(idx)
                    temp_chosen.append(idx)
                    added_count += 1
                    if added_count >= 2:
                        break
            if added_count >= 2:
                best_chosen = temp_chosen
                best_size += added_count
                improved = True
                break
        if not improved:
            break
    
    # Convert back to sorted lists
    return [sorted([i+1 for i in range(16) if all_sets[idx] & (1 << i)]) for idx in best_chosen]
