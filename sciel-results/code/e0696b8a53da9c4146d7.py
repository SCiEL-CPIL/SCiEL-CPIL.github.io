import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    best_size = 0
    best_set = set(prescribed_set)
    
    # Pre-compute sums for prescribed set
    def get_sums(s):
        sums = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums

    initial_sums = get_sums(prescribed_set)
    
    # Helper to check if x can be added to a set with given sums
    def can_add(x, s_set, s_sums):
        if x in s_set:
            return False
        # Check x + y for all y in s_set
        for y in s_set:
            if x + y in s_sums:
                return False
        # Check 2x
        if 2 * x in s_sums:
            return False
        return True

    # Helper to update sums when adding x
    def add_to_sums(x, s_set, s_sums):
        new_sums = set(s_sums)
        for y in s_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    # Local search: try to add elements, then try swaps
    def local_search(current_set, current_sums, max_steps=150):
        steps = 0
        while steps < max_steps and time.time() < deadline:
            steps += 1
            # Try to add any element
            added = False
            candidates_list = list(range(12, 285))
            random.shuffle(candidates_list)
            
            for x in candidates_list:
                if can_add(x, current_set, current_sums):
                    current_set.add(x)
                    current_sums = add_to_sums(x, current_set, current_sums)
                    added = True
                    break
            
            if added:
                continue
            
            # If no element could be added, try a swap
            removable = [x for x in current_set if x not in prescribed_set]
            if not removable:
                break
                
            # Try multiple random removals
            swap_done = False
            for _ in range(5):
                if time.time() >= deadline:
                    break
                x_remove = random.choice(removable)
                
                temp_set = set(current_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums(temp_set)
                
                # Try to find a y to add
                for _ in range(30):
                    if time.time() >= deadline:
                        break
                    y_add = random.choice(candidates_list)
                    if y_add in temp_set:
                        continue
                    if can_add(y_add, temp_set, temp_sums):
                        # Accept swap
                        current_set = temp_set
                        current_set.add(y_add)
                        current_sums = add_to_sums(y_add, current_set, temp_sums)
                        swap_done = True
                        break
                
                if swap_done:
                    break
            
            if not swap_done:
                # Try another approach: remove two and add one? Or just break
                pass

        return current_set, current_sums

    # Main search loop
    # We will do several random greedy restarts and local searches.
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # 1. Random Greedy Construction
        current_set = set(prescribed_set)
        current_sums = get_sums(current_set)
        
        candidates_list = list(range(12, 285))
        random.shuffle(candidates_list)
        
        for x in candidates_list:
            if can_add(x, current_set, current_sums):
                current_set.add(x)
                current_sums = add_to_sums(x, current_set, current_sums)
        
        # 2. Local Search
        current_set, current_sums = local_search(current_set, current_sums, max_steps=150)
        
        # 3. Update Best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # Optional: If we hit a very high size, we might stop?
            # Max is 20. If we hit 20, it's likely optimal.
            if best_size >= 20:
                break

    # Final polish: try all single swaps on the best set
    if time.time() < deadline - 5:
        best_sums = get_sums(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            removable = [x for x in best_set if x not in prescribed_set]
            for x_remove in removable:
                if time.time() >= deadline - 5:
                    break
                temp_set = set(best_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums(temp_set)
                
                # Try all possible additions
                for y_add in range(1, 285):
                    if y_add in temp_set:
                        continue
                    if can_add(y_add, temp_set, temp_sums):
                        # Check if this swap actually improves (it should, since we're adding)
                        new_set = set(temp_set)
                        new_set.add(y_add)
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_sums = get_sums(best_set)
                            improved = True
                            break
                if improved:
                    break

    # Final validation before returning
    final_sums = get_sums(best_set)
    if len(final_sums) != len(best_set) * (len(best_set) + 1) // 2:
        # Fallback to prescribed set if something went wrong
        return sorted(prescribed)
        
    return sorted(list(best_set))
