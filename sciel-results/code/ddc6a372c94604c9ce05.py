import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    D = 1
    for d in denominators:
        D = D * d // gcd(D, d)
        
    # Weights
    weights = [D // d for d in denominators]
    
    # State: list of booleans or bitmask
    # Using a list of indices in the current set for fast sum updates?
    # Or just a bitmask and recompute sum? 
    # Recomputing sum of 47 items is fast.
    # Better: maintain current_sum.
    
    best_sum = 0
    best_mask = 0
    
    # Helper to check validity and update best
    def update_best(mask, s):
        nonlocal best_sum, best_mask
        if s > best_sum:
            best_sum = s
            best_mask = mask
            
    # Initialization: Best Fit Decreasing
    # Sort indices by weight descending
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    current_mask = 0
    current_sum = 0
    in_set = [False] * n
    
    for i in sorted_indices:
        if current_sum + weights[i] < D:
            current_sum += weights[i]
            current_mask |= (1 << i)
            in_set[i] = True
            
    update_best(current_mask, current_sum)
    
    # Local Search
    # We will use a simple best-improvement strategy.
    # To speed up, we can precompute lists of items in and out.
    
    def local_search(mask, s):
        nonlocal best_sum, best_mask
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            in_set = [(mask >> i) & 1 for i in range(n)]
            in_items = [i for i in range(n) if in_set[i]]
            out_items = [i for i in range(n) if not in_set[i]]
            
            # 1. Try Add
            for j in out_items:
                new_s = s + weights[j]
                if new_s < D and new_s > s:
                    s = new_s
                    mask |= (1 << j)
                    improved = True
                    break # Restart loop to re-evaluate with new state
                    
            if improved:
                continue
                
            # 2. Try 1-1 Swap
            for i in in_items:
                for j in out_items:
                    new_s = s - weights[i] + weights[j]
                    if new_s < D and new_s > s:
                        s = new_s
                        mask &= ~(1 << i)
                        mask |= (1 << j)
                        improved = True
                        break
                if improved:
                    break
                    
            if improved:
                continue
                
            # 3. Try 1-2 Swap (Remove 1, Add 2)
            # This is O(N^3). 47^3 ~ 100k. Might be ok for a few iterations.
            for i in in_items:
                base = s - weights[i]
                for j in out_items:
                    new_s1 = base + weights[j]
                    # If new_s1 >= D, adding another positive weight won't help (weights are positive)
                    # So we only need to check k if new_s1 < D
                    for k in out_items:
                        if k == j: continue
                        new_s = new_s1 + weights[k]
                        if new_s < D and new_s > s:
                            s = new_s
                            mask &= ~(1 << i)
                            mask |= (1 << j)
                            mask |= (1 << k)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
            if improved:
                continue
                
            # 4. Try 2-1 Swap (Remove 2, Add 1)
            for i in in_items:
                for k in in_items:
                    if i == k: continue
                    base = s - weights[i] - weights[k]
                    for j in out_items:
                        new_s = base + weights[j]
                        if new_s < D and new_s > s:
                            s = new_s
                            mask &= ~(1 << i)
                            mask &= ~(1 << k)
                            mask |= (1 << j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
            if improved:
                continue
                
            # If no improvement, kick
            # Randomly flip 2-3 bits
            import random
            kick_count = random.randint(2, 4)
            for _ in range(kick_count):
                idx = random.randint(0, n-1)
                mask ^= (1 << idx)
                if (mask >> idx) & 1:
                    s += weights[idx]
                else:
                    s -= weights[idx]
            
            # If kick made it invalid (s >= D), we need to fix it?
            # Or just restart from a valid state?
            # Simpler: If s >= D, revert kick or find a valid state.
            # Actually, if s >= D, we can just remove items until valid.
            if s >= D:
                # Remove largest items until valid
                # This might destroy the "best" property, but it's a kick.
                # Better: just restart the local search from this invalid state?
                # No, local search assumes valid start.
                # Let's just revert the kick if it becomes invalid.
                # Or, simpler: if invalid, just do a greedy repair.
                # For now, let's just try to remove items.
                in_set = [(mask >> i) & 1 for i in range(n)]
                in_items = [i for i in range(n) if in_set[i]]
                in_items.sort(key=lambda x: weights[x], reverse=True)
                for i in in_items:
                    if s < D: break
                    s -= weights[i]
                    mask &= ~(1 << i)
                    
            # Update best if current is better (it shouldn't be after kick, but maybe)
            if s > best_sum:
                best_sum = s
                best_mask = mask

    # Run multiple restarts if time permits?
    # The kick inside local_search handles restarts implicitly.
    # But a full restart from a different heuristic might help.
    
    # Let's just run the main loop.
    local_search(current_mask, current_sum)
    
    # Convert best_mask to indices
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result
