import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    all_masks = [1 << i for i in range(n)]
    
    # Helper to check strict subset
    def is_subset(a, b):
        return (a & b) == a and a != b

    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    # Greedy Initialization
    current_masks = []
    used_masks = set()
    
    # Add empty set first
    current_masks.append(0)
    used_masks.add(0)
    
    for _ in range(m - 1):
        best_mask = -1
        best_score = -1
        # Sample candidates to speed up init? Or check all?
        # 4096 is small. Check all.
        for mask in range(1, 1 << n):
            if mask in used_masks:
                continue
            # Count comparables with existing
            score = 0
            for cm in current_masks:
                if is_comparable(mask, cm):
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        else:
            # Fallback
            break
            
    # Pad if needed (should not happen)
    while len(current_masks) < m:
        mask = random.randint(1, (1 << n) - 1)
        if mask not in used_masks:
            current_masks.append(mask)
            used_masks.add(mask)

    # Calculate initial score
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable(masks[i], masks[j]):
                    s += 1
        return s

    best_score = calc_score(current_masks)
    best_solution = list(current_masks)

    # Local Search
    random.seed(42)
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Pick a random index to modify
        idx = random.randint(0, m - 1)
        old_mask = current_masks[idx]
        
        # Generate new mask
        # Strategy: Mutate old mask or pick random
        if random.random() < 0.5:
            # Mutate
            new_mask = old_mask
            # Add or remove a bit
            if random.random() < 0.5:
                # Remove
                if old_mask > 0:
                    bit = random.choice([i for i in range(n) if (old_mask >> i) & 1])
                    new_mask = old_mask ^ (1 << bit)
            else:
                # Add
                if old_mask != (1 << n) - 1:
                    bit = random.choice([i for i in range(n) if not ((old_mask >> i) & 1)])
                    new_mask = old_mask | (1 << bit)
        else:
            # Random
            new_mask = random.randint(0, (1 << n) - 1)
            
        if new_mask == old_mask:
            continue
            
        if new_mask in used_masks:
            # Check if it's the same as another set
            # We need to ensure distinctness
            # If new_mask is in used_masks, it might be old_mask (handled) or another set.
            # If it's another set, we can't use it directly as it violates distinctness.
            # We could swap, but let's just skip for simplicity in this fast loop.
            continue

        # Calculate delta
        delta = 0
        for j in range(m):
            if j == idx:
                continue
            old_comp = is_comparable(old_mask, current_masks[j])
            new_comp = is_comparable(new_mask, current_masks[j])
            delta += new_comp - old_comp
            
        if delta > 0:
            # Accept
            used_masks.remove(old_mask)
            used_masks.add(new_mask)
            current_masks[idx] = new_mask
            best_score += delta
            best_solution = list(current_masks)
        # Optionally: Simulated Annealing step if delta < 0
        # But greedy is fast and effective here.

    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
