import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute all arithmetic progressions of length >= 2 within [1, 256]
    # Note: Singletons (length 1) always have sum +/- 1. 
    # The max sum will be at least 1.
    # We minimize max over all APs. 
    # If we find a sequence where all APs (len>=2) have sum 0 or 1, the max is 1.
    # If we can make all APs (len>=2) have sum 0, the max is 1 (due to singletons).
    # So effectively we want to minimize max(|sum|) for len>=2, but floor at 1.
    
    ap_indices = []
    pos_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = []
            cur = start
            while cur <= n:
                terms.append(cur - 1) # 0-based
                cur += step
            if len(terms) >= 2:
                ap_indices.append(terms)
                for idx in terms:
                    pos_to_aps[idx].append(ap_id)
                ap_id += 1
                
    num_aps = len(ap_indices)
    
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    def compute_stats(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for j in range(num_aps):
            s = 0
            for idx in ap_indices[j]:
                s += signs[idx]
            sums[j] = s
        abs_sums = np.abs(sums)
        max_sum = int(np.max(abs_sums)) if num_aps > 0 else 0
        # Count of APs with max_sum
        count_max = int(np.sum(abs_sums == max_sum))
        return max_sum, count_max, sums

    def local_search(signs_init, max_iters, deadline):
        signs = signs_init.copy()
        current_max, current_count, sums = compute_stats(signs)
        
        # Histogram of absolute sums
        # Max possible sum is n (256). But realistically much lower.
        # Let's size it to n to be safe, or just track max.
        hist = np.zeros(n + 1, dtype=np.int32)
        for s in sums:
            hist[abs(s)] += 1
            
        best_max = current_max
        best_count = current_count
        best_signs = signs.copy()
        
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            signs[i] = new_sign
            
            old_max = current_max
            old_count = current_count
            
            # Update sums and histogram
            affected = pos_to_aps[i]
            
            # We need to be careful with max updates.
            # If we change an AP from max to non-max, max might drop.
            # If we change an AP to a new max, max might rise.
            
            # To simplify, we can just recalculate max from hist if it changed significantly?
            # No, let's do it properly.
            
            # Store old values to revert if needed? 
            # Or just apply and check.
            
            # Let's apply the change.
            for j in affected:
                old_val = sums[j]
                new_val = old_val - 2 * old_sign
                sums[j] = new_val
                
                old_abs = abs(old_val)
                new_abs = abs(new_val)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
            # Determine new max
            new_max = old_max
            # Check if we created a new max
            # The new values are new_val. The max possible new_abs is max(old_abs, new_abs).
            # Actually, just scan hist from current_max upwards? 
            # Or keep track of potential new max.
            
            # If hist[old_max] became 0, we need to find new max.
            # If any new_abs > old_max, new_max = max(new_abs).
            
            # Efficient way:
            # 1. Check if any new_abs > old_max.
            max_new_abs = 0
            for j in affected:
                new_abs = abs(sums[j])
                if new_abs > max_new_abs:
                    max_new_abs = new_abs
            
            if max_new_abs > old_max:
                new_max = max_new_abs
            else:
                # Max didn't increase.
                # Did it decrease?
                if hist[old_max] == 0:
                    # Find new max
                    new_max = old_max - 1
                    while new_max >= 0 and hist[new_max] == 0:
                        new_max -= 1
                    if new_max < 0:
                        new_max = 0 # Should not happen with singletons logic, but safe
                else:
                    new_max = old_max
                    
            # Count of APs with new_max
            new_count = hist[new_max]
            
            # Acceptance criteria:
            # Minimize (new_max, new_count)
            if new_max < old_max or (new_max == old_max and new_count < old_count):
                # Accept
                current_max = new_max
                current_count = new_count
                
                if current_max < best_max or (current_max == best_max and current_count < best_count):
                    best_max = current_max
                    best_count = current_count
                    best_signs = signs.copy()
                    
                if best_max == 1: # Theoretical min for n=256 (singletons)
                    # Can we get 0? No, singletons are +/-1.
                    # So 1 is the optimal max.
                    break
            else:
                # Revert
                signs[i] = old_sign
                for j in affected:
                    old_val = sums[j] # This is the new_val after flip
                    # We need to revert sums and hist
                    # sums[j] is currently new_val. We want old_val.
                    # new_val = old_val_orig - 2*old_sign
                    # old_val_orig = new_val + 2*old_sign
                    # But we lost old_val_orig in the loop above? 
                    # No, we updated sums[j] in place.
                    # Let's store old_vals before modifying.
                    
                    # This revert logic is getting messy. 
                    # Let's store the changes.
                    pass # See revised code below

        return best_signs, best_max, best_count

    # Revised local search with proper revert
    def local_search_v2(signs_init, max_iters, deadline):
        signs = signs_init.copy()
        current_max, current_count, sums = compute_stats(signs)
        
        hist = np.zeros(n + 1, dtype=np.int32)
        for s in sums:
            hist[abs(s)] += 1
            
        best_max = current_max
        best_count = current_count
        best_signs = signs.copy()
        
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            
            affected = pos_to_aps[i]
            
            # Store old state for revert
            old_sums = {}
            for j in affected:
                old_sums[j] = sums[j]
                
            # Apply flip
            signs[i] = -old_sign
            
            new_max_candidate = 0
            for j in affected:
                old_val = old_sums[j]
                new_val = old_val - 2 * old_sign
                sums[j] = new_val
                
                old_abs = abs(old_val)
                new_abs = abs(new_val)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                if new_abs > new_max_candidate:
                    new_max_candidate = new_abs
            
            # Calculate new max
            old_max = current_max
            if new_max_candidate > old_max:
                new_max = new_max_candidate
            else:
                if hist[old_max] == 0:
                    new_max = old_max - 1
                    while new_max >= 0 and hist[new_max] == 0:
                        new_max -= 1
                    if new_max < 0: new_max = 0
                else:
                    new_max = old_max
                    
            new_count = hist[new_max]
            
            # Acceptance
            if new_max < old_max or (new_max == old_max and new_count < old_count):
                current_max = new_max
                current_count = new_count
                
                if current_max < best_max or (current_max == best_max and current_count < best_count):
                    best_max = current_max
                    best_count = current_count
                    best_signs = signs.copy()
                    
                if best_max == 1:
                    break
            else:
                # Revert
                signs[i] = old_sign
                for j in affected:
                    old_val = old_sums[j]
                    sums[j] = old_val
                    hist[abs(sums[j])] += 1
                    # We need to revert the hist changes made during the flip
                    # The flip did: hist[old_abs]-=1, hist[new_abs]+=1
                    # So revert: hist[old_abs]+=1, hist[new_abs]-=1
                    # But we don't have new_abs stored separately.
                    # We can just recalculate hist from sums? No, too slow.
                    # We can store new_vals too.
                    
        return best_signs, best_max, best_count

    # This revert logic is still fragile. 
    # Better: Store old_abs and new_abs for each affected AP.
    
    def local_search_v3(signs_init, max_iters, deadline):
        signs = signs_init.copy()
        current_max, current_count, sums = compute_stats(signs)
        
        hist = np.zeros(n + 1, dtype=np.int32)
        for s in sums:
            hist[abs(s)] += 1
            
        best_max = current_max
        best_count = current_count
        best_signs = signs.copy()
        
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            
            affected = pos_to_aps[i]
            
            # Store changes
            changes = []
            for j in affected:
                old_val = sums[j]
                new_val = old_val - 2 * old_sign
                changes.append((j, old_val, new_val))
                
            # Apply
            signs[i] = -old_sign
            new_max_candidate = 0
            for j, old_val, new_val in changes:
                sums[j] = new_val
                old_abs = abs(old_val)
                new_abs = abs(new_val)
                hist[old_abs] -= 1
                hist[new_abs] += 1
                if new_abs > new_max_candidate:
                    new_max_candidate = new_abs
            
            # Determine new max
            old_max = current_max
            if new_max_candidate > old_max:
                new_max = new_max_candidate
            else:
                if hist[old_max] == 0:
                    new_max = old_max - 1
                    while new_max >= 0 and hist[new_max] == 0:
                        new_max -= 1
                    if new_max < 0: new_max = 0
                else:
                    new_max = old_max
                    
            new_count = hist[new_max]
            
            # Acceptance
            if new_max < old_max or (new_max == old_max and new_count < old_count):
                current_max = new_max
                current_count = new_count
                
                if current_max < best_max or (current_max == best_max and current_count < best_count):
                    best_max = current_max
                    best_count = current_count
                    best_signs = signs.copy()
                    
                if best_max == 1:
                    break
            else:
                # Revert
                signs[i] = old_sign
                for j, old_val, new_val in changes:
                    sums[j] = old_val
                    old_abs = abs(old_val)
                    new_abs = abs(new_val)
                    hist[new_abs] -= 1
                    hist[old_abs] += 1

        return best_signs, best_max, best_count

    best_overall = None
    best_overall_max = n + 1
    best_overall_count = n + 1

    inits = []
    tm = thue_morse_signs(n)
    inits.append(tm)
    inits.append(-tm)
    
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999]:
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        inits.append(signs)

    for signs_init in inits:
        if time.time() > deadline - 5.0:
            break
        
        signs, mx, cnt = local_search_v3(signs_init, max_iters=5000, deadline=deadline)
        
        if mx < best_overall_max or (mx == best_overall_max and cnt < best_overall_count):
            best_overall_max = mx
            best_overall_count = cnt
            best_overall = signs.copy()
            
            if best_overall_max == 1:
                break

    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
    
    return [int(s) for s in best_overall]
