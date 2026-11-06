import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Split: Small = 2..18, Large = 19..256
    small_nums = list(range(2, 19))
    large_nums = list(range(19, 257))
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Map global index to local index
    # Global index i corresponds to nums[i]
    # small_nums[k] is nums[2 + k - 2] = nums[k]? No.
    # nums[0] = 2, nums[1] = 3, ...
    # small_nums[0] = 2 (global idx 0)
    # small_nums[k] = 2 + k (global idx k)
    # large_nums[0] = 19 (global idx 17)
    # large_nums[k] = 19 + k (global idx 17 + k)
    
    # Precompute conflicts
    # conf_small[i]: bitmask of small conflicts
    conf_small = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(i + 1, n_small):
            b = small_nums[j]
            if a * b // gcd(a, b) <= 256:
                conf_small[i] |= (1 << j)
                conf_small[j] |= (1 << i)
                
    # conf_small_large[i]: bitmask of large conflicts for small i
    conf_small_large = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        mask = 0
        for j in range(n_large):
            b = large_nums[j]
            if a * b // gcd(a, b) <= 256:
                mask |= (1 << j)
        conf_small_large[i] = mask
        
    # conf_large[j]: bitmask of large conflicts for large j
    conf_large = [0] * n_large
    for i in range(n_large):
        a = large_nums[i]
        for j in range(i + 1, n_large):
            b = large_nums[j]
            if a * b // gcd(a, b) <= 256:
                conf_large[i] |= (1 << j)
                conf_large[j] |= (1 << i)
                
    # Weights
    w_small = [1.0 / v for v in small_nums]
    w_large = [1.0 / v for v in large_nums]
    
    # Sort large indices by weight descending
    large_order = sorted(range(n_large), key=lambda x: -w_large[x])
    
    all_large_mask = (1 << n_large) - 1
    
    best_total = 0.0
    best_small_mask = 0
    best_large_mask = 0
    
    # Enumerate small subsets
    for sm in range(1 << n_small):
        if time.time() > deadline - 10:
            break
            
        # Check independence
        valid = True
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if conf_small[i] & sm:
                valid = False
                break
            m ^= lsb
        if not valid:
            continue
            
        # Compute forbidden large mask
        forbidden = 0
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            forbidden |= conf_small_large[i]
            m ^= lsb
            
        available = all_large_mask & ~forbidden
        
        # Greedy on large
        sel_large = 0
        sum_large = 0.0
        for j in large_order:
            if available & (1 << j):
                if not (conf_large[j] & sel_large):
                    sel_large |= (1 << j)
                    sum_large += w_large[j]
                    
        # Sum small
        sum_small = 0.0
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            sum_small += w_small[i]
            m ^= lsb
            
        total = sum_small + sum_large
        if total > best_total:
            best_total = total
            best_small_mask = sm
            best_large_mask = sel_large
            
    # Reconstruct best set
    best_set = set()
    m = best_small_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        best_set.add(small_nums[i])
        m ^= lsb
        
    m = best_large_mask
    while m:
        lsb = m & -m
        j = lsb.bit_length() - 1
        best_set.add(large_nums[j])
        m ^= lsb
        
    # Local Search Polish
    # Map number to index in global list for conflict checks?
    # Or just use the precomputed structures?
    # It's easier to rebuild a conflict dict for the specific set or use global indices.
    
    # Let's use global indices for the final polish to be safe and simple.
    # n = 255 (2 to 256)
    # conflict_global[i] is set of global indices conflicting with i.
    
    conflict_global = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] * nums[j] // gcd(nums[i], nums[j]) <= 256:
                conflict_global[i].add(j)
                conflict_global[j].add(i)
                
    # Convert best_set to set of global indices
    num_to_idx = {v: i for i, v in enumerate(nums)}
    selected_idx = {num_to_idx[v] for v in best_set}
    
    # Local search
    for _ in range(20): # Iterations
        if time.time() > deadline - 2:
            break
        improved = False
        
        # 1. Add
        for j in range(n):
            if j not in selected_idx and not (conflict_global[j] & selected_idx):
                selected_idx.add(j)
                improved = True
                break
        if improved: continue
        
        # 2. 1-for-1 swap
        for i in list(selected_idx):
            s_without = selected_idx - {i}
            base_sum = sum(1.0 / nums[i] for i in s_without)
            for j in range(n):
                if j not in s_without and not (conflict_global[j] & s_without):
                    new_sum = base_sum + 1.0 / nums[j]
                    cur_sum = sum(1.0 / nums[i] for i in selected_idx)
                    if new_sum > cur_sum + 1e-15:
                        selected_idx = s_without | {j}
                        improved = True
                        break
            if improved: break
        if improved: continue
        
        # 3. 1-for-2 swap (Remove 1, Add 2)
        for i in list(selected_idx):
            s_without = selected_idx - {i}
            cur_sum = sum(1.0 / nums[i] for i in selected_idx)
            addable = [j for j in range(n) if j not in s_without and not (conflict_global[j] & s_without)]
            for a in range(len(addable)):
                for b in range(a + 1, len(addable)):
                    j, k = addable[a], addable[b]
                    if not (conflict_global[j] & conflict_global[k]):
                        new_sum = sum(1.0 / nums[x] for x in s_without) + 1.0/nums[j] + 1.0/nums[k]
                        if new_sum > cur_sum + 1e-15:
                            selected_idx = s_without | {j, k}
                            improved = True
                            break
                if improved: break
            if improved: break
        if improved: continue
        
        # 4. 2-for-1 swap (Remove 2, Add 1)
        sel_list = list(selected_idx)
        for a in range(len(sel_list)):
            for b in range(a + 1, len(sel_list)):
                i, j = sel_list[a], sel_list[b]
                s_without = selected_idx - {i, j}
                cur_sum = sum(1.0 / nums[x] for x in selected_idx)
                for k in range(n):
                    if k not in s_without and not (conflict_global[k] & s_without):
                        new_sum = sum(1.0 / nums[x] for x in s_without) + 1.0/nums[k]
                        if new_sum > cur_sum + 1e-15:
                            selected_idx = s_without | {k}
                            improved = True
                            break
                if improved: break
            if improved: break
        if not improved:
            break
            
    # Final result
    result = sorted(nums[i] for i in selected_idx)
    if not result:
        return [2]
    return result
