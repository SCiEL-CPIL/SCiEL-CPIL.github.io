import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability bitsets
    # comp[i] is an integer where bit j is 1 if mask i and mask j are comparable (i != j)
    comp = [0] * total_masks
    
    # Optimization: Use symmetry or direct calculation.
    # Direct calculation is O(N^2). N=4096. N^2 = 16.7M.
    # This is fast enough in Python if we keep it tight.
    
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i != j:
                # Check if i is subset of j or j is subset of i
                if (i & j) == i or (i & j) == j:
                    ci |= (1 << j)
        comp[i] = ci

    # Greedy Initialization
    selected = []
    selected_mask = 0
    all_masks = list(range(total_masks))
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        # To speed up, we could randomize the order or just scan all.
        # Scanning all 4096 is fine.
        for mask in all_masks:
            if (selected_mask >> mask) & 1:
                continue
            score = (comp[mask] & selected_mask).bit_count()
            if score > best_score:
                best_score = score
                best_mask = mask
        
        selected.append(best_mask)
        selected_mask |= (1 << best_mask)
    
    # Calculate initial score
    current_score = 0
    for i in range(m):
        current_score += (comp[selected[i]] & selected_mask).bit_count()
    current_score //= 2
    
    best_score = current_score
    best_selection = list(selected)
    
    # Simulated Annealing
    temperature = 10.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    # Pre-compute list of indices for random selection
    indices = list(range(m))
    
    while time.time() < deadline:
        if temperature < min_temp:
            break
        
        # 10% chance of 2-swap, 90% chance of 1-swap
        if random.random() < 0.1:
            # 2-Swap Move
            idx1, idx2 = random.sample(indices, 2)
            old1 = selected[idx1]
            old2 = selected[idx2]
            
            sel_without_both = selected_mask & ~(1 << old1) & ~(1 << old2)
            
            # Loss
            loss1 = (comp[old1] & sel_without_both).bit_count()
            loss2 = (comp[old2] & sel_without_both).bit_count()
            # Check if old1 and old2 are comparable
            if (comp[old1] >> old2) & 1:
                total_loss = loss1 + loss2 + 1
            else:
                total_loss = loss1 + loss2
            
            # Pick two new masks
            new1 = -1
            new2 = -1
            attempts = 0
            while (new1 == -1 or new2 == -1) and attempts < 100:
                cand = random.randint(0, total_masks - 1)
                if not ((selected_mask >> cand) & 1) and cand != old1 and cand != old2:
                    if new1 == -1:
                        new1 = cand
                    elif cand != new1:
                        new2 = cand
                attempts += 1
            
            if new1 != -1 and new2 != -1:
                # Gain
                gain1 = (comp[new1] & sel_without_both).bit_count()
                gain2 = (comp[new2] & sel_without_both).bit_count()
                if (comp[new1] >> new2) & 1:
                    total_gain = gain1 + gain2 + 1
                else:
                    total_gain = gain1 + gain2
                
                delta = total_gain - total_loss
                
                if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1)):
                    selected[idx1] = new1
                    selected[idx2] = new2
                    selected_mask = sel_without_both | (1 << new1) | (1 << new2)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected)
        else:
            # 1-Swap Move
            idx = random.randint(0, m - 1)
            old_mask = selected[idx]
            
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp[old_mask] & sel_without_old).bit_count()
            
            new_mask = -1
            attempts = 0
            while new_mask == -1 and attempts < 100:
                cand = random.randint(0, total_masks - 1)
                if not ((selected_mask >> cand) & 1):
                    new_mask = cand
                attempts += 1
            
            if new_mask != -1:
                gain = (comp[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                
                if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1)):
                    selected[idx] = new_mask
                    selected_mask = sel_without_old | (1 << new_mask)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected)
        
        temperature *= cooling_rate

    # Final Polish: Hill Climbing (1-swap)
    if time.time() < deadline:
        selected = list(best_selection)
        selected_mask = 0
        for s in selected:
            selected_mask |= (1 << s)
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(m):
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp[old_mask] & sel_without_old).bit_count()
                
                best_new = -1
                best_delta = 0
                for cand in all_masks:
                    if (selected_mask >> cand) & 1:
                        continue
                    gain = (comp[cand] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = cand
                
                if best_new != -1 and best_delta > 0:
                    selected[idx] = best_new
                    selected_mask = sel_without_old | (1 << best_new)
                    current_score += best_delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected)
        
        # One more pass of 2-swap hill climbing?
        # Might be too slow. Stick to 1-swap polish.

    # Construct final result
    result = []
    for mask in best_selection:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
