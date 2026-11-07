import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability
    # comp[i] is bitset of j such that i and j are comparable
    comp = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i != j:
                if (i & j) == i or (i & j) == j:
                    ci |= (1 << j)
        comp[i] = ci

    def generate_chain(order):
        """Generate a chain by adding bits in the given order."""
        chain = [0]
        mask = 0
        for bit in order:
            mask |= (1 << bit)
            chain.append(mask)
        return chain

    # Initialize with 4 chains based on different permutations
    # We need 40 distinct sets.
    # Chain 1: 0,1,2,3,4,5,6,7,8,9,10,11 (13 sets)
    # Chain 2: 1,2,3,4,5,6,7,8,9,10,11,0 (13 sets, but last is same as Chain 1 last)
    # We need to ensure distinctness.
    
    selected = set()
    # Add Chain 1
    c1 = generate_chain([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    for s in c1:
        selected.add(s)
        
    # Add Chain 2 (avoiding duplicates)
    c2 = generate_chain([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])
    for s in c2:
        if s not in selected:
            selected.add(s)
            
    # Add Chain 3
    c3 = generate_chain([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1])
    for s in c3:
        if s not in selected:
            selected.add(s)
            
    # Add Chain 4
    c4 = generate_chain([3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2])
    for s in c4:
        if s not in selected:
            selected.add(s)
            
    # If we don't have 40, add random sets
    if len(selected) < m:
        for mask in range(total_masks):
            if mask not in selected:
                selected.add(mask)
                if len(selected) == m:
                    break
                    
    # If we have more than 40, remove least useful ones
    if len(selected) > m:
        selected_list = list(selected)
        # Sort by degree in comparability graph (within selected)
        # This is a heuristic to keep the "core"
        # For now, just take the first 40
        selected = set(selected_list[:m])
        
    selected_list = list(selected)
    selected_mask = 0
    for s in selected_list:
        selected_mask |= (1 << s)
        
    # Calculate score
    def calc_score(sel_list):
        sm = 0
        for s in sel_list:
            sm |= (1 << s)
        score = 0
        for s in sel_list:
            score += (comp[s] & sm).bit_count()
        return score // 2

    current_score = calc_score(selected_list)
    best_score = current_score
    best_selection = list(selected_list)
    
    # Local Search: Hill Climbing
    # We try to swap one element with another to improve score.
    # To be fast, we precompute the gain/loss for each potential swap.
    
    indices = list(range(m))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in indices:
            old_mask = selected_list[idx]
            sel_without_old_mask = selected_mask & ~(1 << old_mask)
            loss = (comp[old_mask] & sel_without_old_mask).bit_count()
            
            best_new = -1
            best_delta = 0
            
            # Scan all possible new masks
            for cand in range(total_masks):
                if (selected_mask >> cand) & 1:
                    continue
                gain = (comp[cand] & sel_without_old_mask).bit_count()
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_new = cand
            
            if best_new != -1 and best_delta > 0:
                selected_list[idx] = best_new
                selected_mask = sel_without_old_mask | (1 << best_new)
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selection = list(selected_list)
                    
        # If no improvement in 1-swap, try 2-swap occasionally?
        # 1-swap is often sufficient for this structure.
        # Add a random restart if stuck?
        # For now, stick to hill climbing.
        
    # Final polish: try to improve with random perturbations if time remains
    if time.time() < deadline - 5.0:
        temp = 5.0
        while time.time() < deadline - 1.0 and temp > 0.1:
            idx = random.randint(0, m - 1)
            old_mask = selected_list[idx]
            sel_without_old_mask = selected_mask & ~(1 << old_mask)
            loss = (comp[old_mask] & sel_without_old_mask).bit_count()
            
            cand = random.randint(0, total_masks - 1)
            if (selected_mask >> cand) & 1:
                continue
                
            gain = (comp[cand] & sel_without_old_mask).bit_count()
            delta = gain - loss
            
            if delta > 0 or random.random() < math.exp(delta / temp):
                selected_list[idx] = cand
                selected_mask = sel_without_old_mask | (1 << cand)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selection = list(selected_list)
            
            temp *= 0.99

    # Construct result
    result = []
    for mask in best_selection:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
