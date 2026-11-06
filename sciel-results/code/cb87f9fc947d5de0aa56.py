import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    def can_add(idx, mask):
        return (conflict_mask[idx] & mask) == 0

    def get_size(mask):
        return bin(mask).count('1')

    # Phase 1: Greedy Add Evens
    changed = True
    while changed:
        changed = False
        for e in evens_indices:
            if (current_mask >> e) & 1:
                continue
            if can_add(e, current_mask):
                current_mask |= (1 << e)
                changed = True
                sz = get_size(current_mask)
                if sz > best_size:
                    best_mask = current_mask
                    best_size = sz
        if time.time() > DEADLINE - 5:
            break

    # Phase 2: Improved local search with better move selection
    all_indices = list(range(N))
    
    # Precompute for each vertex, which other vertices it conflicts with
    # Sort candidates by degree for better search
    
    max_iters = 300000
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - 2:
            break
        
        r = random.random()
        
        if r < 0.25:
            # Try to add a random even that doesn't conflict
            e = random.choice(evens_indices)
            if (current_mask >> e) & 1:
                continue
            if can_add(e, current_mask):
                current_mask |= (1 << e)
                sz = get_size(current_mask)
                if sz > best_size:
                    best_mask = current_mask
                    best_size = sz
        
        elif r < 0.5:
            # Try 1-swap: Remove one, add one
            to_remove = random.choice(all_indices)
            if not ((current_mask >> to_remove) & 1):
                continue
            trial_mask = current_mask & ~(1 << to_remove)
            
            # Try all evens not in trial_mask
            for e in evens_indices:
                if (trial_mask >> e) & 1:
                    continue
                if can_add(e, trial_mask):
                    new_mask = trial_mask | (1 << e)
                    # Size-neutral, accept with probability
                    if random.random() < 0.15:
                        current_mask = new_mask
                        break
        
        elif r < 0.75:
            # Try 1-Remove, 2-Add (size +1)
            to_remove = random.choice(all_indices)
            if not ((current_mask >> to_remove) & 1):
                continue
            trial_mask = current_mask & ~(1 << to_remove)
            
            # Try pairs of evens
            e1 = random.choice(evens_indices)
            e2 = random.choice(evens_indices)
            if e1 == e2:
                continue
            if (trial_mask >> e1) & 1 or (trial_mask >> e2) & 1:
                continue
            
            if can_add(e1, trial_mask) and can_add(e2, trial_mask | (1 << e1)):
                new_mask = trial_mask | (1 << e1) | (1 << e2)
                sz = get_size(new_mask)
                if sz > get_size(current_mask):
                    current_mask = new_mask
                    if sz > best_size:
                        best_mask = current_mask
                        best_size = sz
        else:
            # Try 2-Remove, 3-Add (size +1)
            to_remove1 = random.choice(all_indices)
            to_remove2 = random.choice(all_indices)
            if to_remove1 == to_remove2:
                continue
            if not ((current_mask >> to_remove1) & 1) or not ((current_mask >> to_remove2) & 1):
                continue
            
            trial_mask = current_mask & ~(1 << to_remove1) & ~(1 << to_remove2)
            
            e1 = random.choice(evens_indices)
            e2 = random.choice(evens_indices)
            e3 = random.choice(evens_indices)
            if e1 == e2 or e2 == e3 or e1 == e3:
                continue
            if (trial_mask >> e1) & 1 or (trial_mask >> e2) & 1 or (trial_mask >> e3) & 1:
                continue
            
            if (can_add(e1, trial_mask) and can_add(e2, trial_mask | (1 << e1)) and 
                can_add(e3, trial_mask | (1 << e1) | (1 << e2))):
                new_mask = trial_mask | (1 << e1) | (1 << e2) | (1 << e3)
                sz = get_size(new_mask)
                if sz > get_size(current_mask):
                    current_mask = new_mask
                    if sz > best_size:
                        best_mask = current_mask
                        best_size = sz

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result
