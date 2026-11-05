import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    # Bitmask: sum s represented as bit (s-2). Sums range from 2 to 568.
    def compute_sums_bitmask(arr):
        mask = 0
        n = len(arr)
        for i in range(n):
            a = arr[i]
            for j in range(i, n):
                s = a + arr[j]
                mask |= (1 << (s - 2))
        return mask

    def can_add_bitmask(candidate, arr, mask):
        for y in arr:
            s = candidate + y
            if (mask >> (s - 2)) & 1:
                return False
        s = 2 * candidate
        if (mask >> (s - 2)) & 1:
            return False
        return True

    def add_to_bitmask(candidate, arr, mask):
        new_mask = mask
        for y in arr:
            s = candidate + y
            new_mask |= (1 << (s - 2))
        s = 2 * candidate
        new_mask |= (1 << (s - 2))
        return new_mask

    base_arr = sorted(prescribed)
    base_mask = compute_sums_bitmask(base_arr)
    
    best_set = base_arr[:]
    best_mask = base_mask
    best_size = len(best_set)

    def greedy_fill(arr, mask, candidates, time_limit):
        local_arr = arr[:]
        local_mask = mask
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in local_arr:
                continue
            if can_add_bitmask(x, local_arr, local_mask):
                local_arr.append(x)
                local_mask = add_to_bitmask(x, local_arr[:-1], local_mask)
        return local_arr, local_mask

    # Initial greedy fill
    candidates = list(range(12, 285))
    greedy_arr, greedy_mask = greedy_fill(base_arr, base_mask, candidates, deadline)
    if len(greedy_arr) > best_size:
        best_size = len(greedy_arr)
        best_set = greedy_arr[:]
        best_mask = greedy_mask

    rng = random.Random(42)
    
    # Phase 1: Randomized greedy restarts with perturbation
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Perturb: remove 1-2 random non-prescribed elements
        removable = [x for x in best_set if x not in prescribed_set]
        if not removable:
            continue
        
        num_remove = min(rng.randint(1, 2), len(removable))
        to_remove = rng.sample(removable, num_remove)
        start_set = [x for x in best_set if x not in to_remove]
        start_mask = compute_sums_bitmask(start_set)
        
        # Shuffle candidates for diversity
        shuffled = list(range(1, 285))
        rng.shuffle(shuffled)
        
        new_arr, new_mask = greedy_fill(start_set, start_mask, shuffled, deadline)
        
        if len(new_arr) > best_size:
            best_size = len(new_arr)
            best_set = new_arr[:]
            best_mask = new_mask
            if best_size >= 20:
                break

    # Phase 2: Local search - try to add elements, then swap if stuck
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        current_set = best_set[:]
        current_mask = best_mask
        
        # Try to add any element
        added = False
        candidates_list = list(range(12, 285))
        rng.shuffle(candidates_list)
        
        for x in candidates_list:
            if x in current_set:
                continue
            if can_add_bitmask(x, current_set, current_mask):
                current_set.append(x)
                current_mask = add_to_bitmask(x, current_set[:-1], current_mask)
                added = True
                break
        
        if added:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                best_mask = current_mask
                if best_size >= 20:
                    break
            continue
        
        # No element could be added - try a swap
        removable = [x for x in current_set if x not in prescribed_set]
        if not removable:
            break
        
        x_remove = rng.choice(removable)
        temp_set = [x for x in current_set if x != x_remove]
        temp_mask = compute_sums_bitmask(temp_set)
        
        # Try random y to add
        found_swap = False
        for _ in range(30):
            y_add = rng.randint(1, 284)
            if y_add in temp_set:
                continue
            if can_add_bitmask(y_add, temp_set, temp_mask):
                new_set = temp_set + [y_add]
                new_mask = add_to_bitmask(y_add, temp_set, temp_mask)
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
                    best_mask = new_mask
                found_swap = True
                break
        
        if not found_swap:
            # Try a different removal
            pass

    # Final validation
    final_sums = set()
    lst = sorted(best_set)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            final_sums.add(lst[i] + lst[j])
    
    expected = len(lst) * (len(lst) + 1) // 2
    if len(final_sums) != expected:
        return sorted(prescribed)
    
    return lst
