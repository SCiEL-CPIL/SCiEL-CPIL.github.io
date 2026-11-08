import random
import math
import time

def score_set(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Calculate U+U and U-U
    # Using sets for simplicity, N is small
    sum_set = set()
    diff_set = set()
    
    # Optimization: sort U first? Not strictly needed for set logic.
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 0.0
        
    score = 1 + log_ratio / log_denom
    return score

def construct():
    # Initialize with a few diverse candidates
    candidates = [
        [0, 1, 2, 3, 7],       # Current seed
        [0, 1, 2, 4, 8, 16, 32, 64],   # Powers of 2
        [0, 1, 3, 9, 27, 81],  # Powers of 3
        [0, 1, 3, 4, 10, 12, 20, 22],  # Random-ish
        [0, 1, 2, 5, 10, 20, 40, 80]   # Geometric
    ]
    
    best_set = candidates[0]
    best_score = score_set(best_set)
    
    # Simulated Annealing Parameters
    max_val = 100
    max_size = 12
    start_temp = 1.0
    cool_rate = 0.995
    start_time = time.time()
    time_limit = 110 # Use most of the available time
    
    current_set = list(best_set)
    current_score = best_score
    temp = start_temp
    
    no_improve_count = 0
    restart_interval = 5000
    
    while time.time() - start_time < time_limit:
        # Generate move
        move_type = random.random()
        
        if move_type < 0.2:
            # Toggle add/remove
            if len(current_set) < max_size:
                # Add random element
                if random.random() < 0.5:
                    # Try to add small number
                    idx = random.randint(1, max_val // 2)
                else:
                    idx = random.randint(1, max_val)
                if idx not in current_set:
                    new_set = current_set + [idx]
                    new_set.sort()
                else:
                    new_set = current_set
            else:
                # Remove random element (keep 0)
                if len(current_set) > 1:
                    idx = random.choice([x for x in current_set if x != 0])
                    new_set = [x for x in current_set if x != idx]
                else:
                    new_set = current_set
        elif move_type < 0.5:
            # Swap
            if len(current_set) > 1:
                # Pick an element to swap (not 0)
                removable = [x for x in current_set if x != 0]
                if removable:
                    old_val = random.choice(removable)
                    new_val = random.randint(1, max_val)
                    if new_val not in current_set:
                        new_set = [x if x != old_val else new_val for x in current_set]
                        new_set.sort()
                    else:
                        new_set = current_set
                else:
                    new_set = current_set
            else:
                new_set = current_set
        elif move_type < 0.8:
            # Local tweak: change one element to a nearby value
            if len(current_set) > 1:
                removable = [x for x in current_set if x != 0]
                if removable:
                    old_val = random.choice(removable)
                    delta = random.randint(-10, 10)
                    new_val = max(1, old_val + delta) # Ensure positive
                    if new_val != old_val and new_val not in current_set:
                        new_set = [x if x != old_val else new_val for x in current_set]
                        new_set.sort()
                    else:
                        new_set = current_set
                else:
                    new_set = current_set
            else:
                new_set = current_set
        else:
            # Jump: Replace 2-3 elements with random ones
            if len(current_set) > 2:
                removable = [x for x in current_set if x != 0]
                k = random.randint(1, min(3, len(removable)))
                to_replace = random.sample(removable, k)
                new_elements = [random.randint(1, max_val) for _ in range(k)]
                # Ensure uniqueness and no 0
                valid_new = [e for e in new_elements if e > 0 and e not in current_set]
                if len(valid_new) == k:
                    new_set = [x for x in current_set if x not in to_replace] + valid_new
                    new_set.sort()
                else:
                    new_set = current_set
            else:
                new_set = current_set
        
        if new_set == current_set:
            continue
            
        new_score = score_set(new_set)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_set = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Cooling
        temp *= cool_rate
        if temp < 1e-6:
            temp = 1e-6
            
        # Restart if stuck
        if no_improve_count > restart_interval:
            # Reset to a random candidate or best so far with perturbation
            if random.random() < 0.3:
                current_set = list(best_set)
            else:
                # Generate a new random sparse set
                size = random.randint(4, max_size)
                # Ensure 0 is included
                others = random.sample(range(1, max_val + 1), size - 1)
                current_set = sorted([0] + others)
            current_score = score_set(current_set)
            no_improve_count = 0
            # Reheat slightly
            temp = max(temp, 0.8)

    return best_set
