import random
import time
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n = len(all_sets)  # 1820
    
    best_size = 0
    best_family = []
    
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for overhead
    
    # Helper to check if c can be added to chosen
    # Returns index of a blocker in chosen if c cannot be added, else -1
    def get_blocker(c, chosen):
        buckets = {}
        for idx_a, a in enumerate(chosen):
            inter = a & c
            if inter not in buckets:
                buckets[inter] = []
            buckets[inter].append((a, idx_a))
        
        for y, group in buckets.items():
            if len(group) < 2:
                continue
            g_len = len(group)
            for i in range(g_len):
                ai, idx_ai = group[i]
                ai_out = ai & ~y
                for j in range(i + 1, g_len):
                    aj, idx_aj = group[j]
                    aj_out = aj & ~y
                    if (ai_out & aj_out) == 0:
                        return idx_ai
        return -1

    def can_add(c, chosen):
        return get_blocker(c, chosen) == -1

    # Helper to convert mask to list
    def mask_to_list(mask):
        return [i + 1 for i in range(16) if mask & (1 << i)]
    
    # Main search loop
    # We will run multiple iterations of "Greedy + Conflict-Driven Local Search"
    
    while time.time() - start_time < time_limit:
        # 1. Greedy Construction
        # Use a different random seed for each iteration to vary the order
        current_seed = int(time.time() * 1000000) % (2**32)
        random.seed(current_seed)
        indices = list(range(n))
        random.shuffle(indices)
        
        chosen = []
        chosen_set = set() # For O(1) lookup
        
        # Greedy Pass
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            c = all_sets[idx]
            if c in chosen_set:
                continue
            if can_add(c, chosen):
                chosen.append(c)
                chosen_set.add(c)
        
        # 2. Local Search (Conflict-Driven Swaps)
        improved = True
        # Limit iterations to prevent infinite loops or excessive time in one restart
        local_search_iter = 0
        max_local_search_iter = 50
        
        while improved and local_search_iter < max_local_search_iter:
            improved = False
            local_search_iter += 1
            if time.time() - start_time > time_limit:
                break
            
            # Build list of candidates not in chosen
            # This is O(N)
            candidates = [i for i in range(n) if all_sets[i] not in chosen_set]
            random.shuffle(candidates)
            
            for idx in candidates:
                if time.time() - start_time > time_limit:
                    break
                
                c = all_sets[idx]
                
                # Check if c can be added
                blocker_idx = get_blocker(c, chosen)
                
                if blocker_idx == -1:
                    # Can add directly
                    chosen.append(c)
                    chosen_set.add(c)
                    improved = True
                else:
                    # Try to swap out the blocker
                    # Save state
                    rem_mask = chosen[blocker_idx]
                    
                    # Remove blocker
                    chosen.pop(blocker_idx)
                    chosen_set.remove(rem_mask)
                    
                    # Check if c can be added now
                    if get_blocker(c, chosen) == -1:
                        # Success
                        chosen.append(c)
                        chosen_set.add(c)
                        improved = True
                    else:
                        # Fail, restore
                        chosen.insert(blocker_idx, rem_mask)
                        chosen_set.add(rem_mask)
                        # Do not mark as improved
                        # We can continue to next candidate or break?
                        # Continuing allows trying other candidates that might work.
        
        # Update best
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
    # Convert best_family to list of lists
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
    
    return result
