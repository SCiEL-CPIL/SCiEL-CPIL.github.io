import time
import random

N = 512

# Precompute masks for each element
# masks[x] is a list of integers. Each integer is a bitmask of the other 3 elements in a 4-AP with x.
masks = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    max_d = (N - a) // 3
    for d in range(1, max_d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # AP: (a, b, c, e)
        # For a: others are b, c, e
        mask_a = (1 << b) | (1 << c) | (1 << e)
        masks[a].append(mask_a)
        
        # For b: others are a, c, e
        mask_b = (1 << a) | (1 << c) | (1 << e)
        masks[b].append(mask_b)
        
        # For c: others are a, b, e
        mask_c = (1 << a) | (1 << b) | (1 << e)
        masks[c].append(mask_c)
        
        # For e: others are a, b, c
        mask_e = (1 << a) | (1 << b) | (1 << c)
        masks[e].append(mask_e)

def can_add_mask(x, S_mask):
    """
    Check if x can be added to set represented by S_mask without creating a 4-AP.
    Returns True if safe.
    """
    for m in masks[x]:
        if (S_mask & m) == m:
            return False
    return True

def greedy_fill(S_mask):
    """
    Try to add any element from 1 to N that can be added to S_mask.
    Returns the updated mask.
    """
    for x in range(1, N + 1):
        if not (S_mask & (1 << x)):
            if can_add_mask(x, S_mask):
                S_mask |= (1 << x)
    return S_mask

def construct():
    start_time = time.time()
    # Keep a safety margin of 10 seconds
    deadline = start_time + 50.0
    
    best_mask = 0
    best_size = 0
    current_mask = 0
    current_size = 0

    # Perform a few initial greedy constructions to find a good starting point
    for _ in range(10):
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        temp_mask = 0
        for x in candidates:
            if can_add_mask(x, temp_mask):
                temp_mask |= (1 << x)
        
        # Final fill
        temp_mask = greedy_fill(temp_mask)
        temp_size = bin(temp_mask).count('1')
        
        if temp_size > best_size:
            best_size = temp_size
            best_mask = temp_mask

    # Initialize current state with the best found
    current_mask = best_mask
    current_size = best_size

    # Perturbation loop: Remove random elements, then greedy fill
    # This allows escaping local optima
    while time.time() < deadline:
        # Decide how many elements to remove
        # Bias towards 1, but occasionally remove more to jump basins
        r = random.random()
        if r < 0.7:
            num_remove = 1
        elif r < 0.9:
            num_remove = 2
        else:
            num_remove = 3
        
        # Get list of elements in current set
        elements_in_set = []
        for i in range(1, N + 1):
            if current_mask & (1 << i):
                elements_in_set.append(i)
                
        if not elements_in_set:
            break
            
        # Pick elements to remove
        if num_remove > len(elements_in_set):
            num_remove = len(elements_in_set)
        
        removed = random.sample(elements_in_set, num_remove)
        
        temp_mask = current_mask
        for x in removed:
            temp_mask &= ~(1 << x)
            
        # Greedy fill to recover size
        new_mask = greedy_fill(temp_mask)
        new_size = bin(new_mask).count('1')
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask
            
        # Update current state to the new local optimum
        current_mask = new_mask
        current_size = new_size

    # Convert best_mask to sorted list
    best_set = [i for i in range(1, N + 1) if (best_mask & (1 << i))]
    return best_set
