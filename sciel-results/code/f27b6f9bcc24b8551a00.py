import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 4096
    target_size = 40  # Theoretical max is ~41, aim for 40
    
    def get_sums_involving(x, selected):
        """
        Returns a set of all triple sums that involve x,
        given the current selected set.
        Assumes x is in selected.
        """
        sums_inv = set()
        # (x, x, x)
        sums_inv.add(3*x)
        
        for y in selected:
            if y == x:
                continue
            # (x, x, y)
            sums_inv.add(2*x + y)
            # (x, y, y)
            sums_inv.add(x + 2*y)
            # (x, y, z) where z is in selected, z != x, z != y
            # We iterate over all z in selected
            for z in selected:
                if z == x or z == y:
                    continue
                # Ensure unordered triple: x <= y <= z
                # But since we are just collecting sums, order doesn't matter for the value.
                # However, we must avoid duplicates in the set construction if we were building it,
                # but here we are just listing sums that *will* be removed.
                # The sum x+y+z is unique for the multiset {x,y,z}.
                # We just need to add it to the set to be removed.
                sums_inv.add(x + y + z)
        return sums_inv

    def is_valid_addition(x, selected, sums):
        """
        Check if x can be added. Returns True if valid, and the new sums.
        """
        new_sums = set()
        
        # (x, x, x)
        s = 3*x
        if s in sums:
            return False, None
        new_sums.add(s)
        
        for y in selected:
            # (x, x, y)
            s = 2*x + y
            if s in sums or s in new_sums:
                return False, None
            new_sums.add(s)
            
            # (x, y, y)
            s = x + 2*y
            if s in sums or s in new_sums:
                return False, None
            new_sums.add(s)
            
        for i in range(len(selected)):
            y = selected[i]
            for j in range(i+1, len(selected)):
                z = selected[j]
                s = x + y + z
                if s in sums or s in new_sums:
                    return False, None
                new_sums.add(s)
                
        return True, new_sums

    def greedy_build(order):
        selected = []
        sums = set()
        for x in order:
            valid, new_sums = is_valid_addition(x, selected, sums)
            if valid:
                selected.append(x)
                sums.update(new_sums)
        return selected, sums

    best_set = []
    best_sums = set()
    best_size = 0
    
    # 1. Initial Greedy: Standard
    standard_order = list(range(1, n+1))
    s, sums = greedy_build(standard_order)
    best_set = s
    best_sums = sums
    best_size = len(s)
    
    # 2. Randomized Greedy Restarts
    np.random.seed(42)
    max_restarts = 300 # Increased restarts
    for i in range(max_restarts):
        if time.time() > deadline - 30: # Leave 30s for local search
            break
        order = np.random.permutation(n) + 1
        s, sums = greedy_build(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sums = sums
    
    # 3. Local Search
    current_set = list(best_set)
    current_sums = set(best_sums)
    current_size = best_size
    
    # Helper to update best
    def update_best(set_, sums_, size_):
        nonlocal best_set, best_sums, best_size
        if size_ > best_size:
            best_size = size_
            best_set = set_
            best_sums = sums_

    update_best(current_set, current_sums, current_size)
    
    # Local Search Loop
    # We will try to improve the set.
    # Strategies:
    # 1. Try to add a new element (size +1)
    # 2. Try to swap an element (size same, change structure)
    # 3. Try to remove 1 and add 2 (size +1)
    
    max_iterations = 5000
    no_improvement_count = 0
    improvement_threshold = 500
    
    while time.time() < deadline - 2 and no_improvement_count < improvement_threshold:
        if current_size >= 40:
            break
            
        # Randomly choose a move type
        # Bias towards "grow" moves if we are far from target
        if np.random.rand() < 0.7:
            # Try to grow: Remove y, add x1, x2
            if not current_set:
                continue
            idx = np.random.randint(len(current_set))
            y_to_remove = current_set[idx]
            
            # Remove y_to_remove
            sums_to_remove = get_sums_involving(y_to_remove, current_set)
            current_sums.difference_update(sums_to_remove)
            current_set.pop(idx)
            current_size -= 1
            
            # Try to add x1
            found_x1 = False
            # Try 20 random candidates for x1
            candidates_x1 = np.random.choice(n, 20, replace=False) + 1
            for x1 in candidates_x1:
                if x1 in current_set:
                    continue
                valid1, sums1 = is_valid_addition(x1, current_set, current_sums)
                if valid1:
                    found_x1 = True
                    # Tentatively add x1
                    current_set.append(x1)
                    current_sums.update(sums1)
                    current_size += 1
                    
                    # Try to add x2
                    found_x2 = False
                    candidates_x2 = np.random.choice(n, 20, replace=False) + 1
                    for x2 in candidates_x2:
                        if x2 in current_set or x2 == x1:
                            continue
                        valid2, sums2 = is_valid_addition(x2, current_set, current_sums)
                        if valid2:
                            found_x2 = True
                            current_set.append(x2)
                            current_sums.update(sums2)
                            current_size += 1
                            no_improvement_count = 0
                            update_best(current_set, current_sums, current_size)
                            break
                    
                    if not found_x2:
                        # Revert x1? Or keep it as a swap?
                        # Keeping it as a swap (remove y, add x1) is a valid move that changes structure.
                        # But we wanted to grow. If we can't grow, we might as well keep the swap
                        # to explore different local minima.
                        # However, if we revert, we are back to original.
                        # Let's keep the swap (size same) to shuffle.
                        # This effectively makes it a "swap" move if "grow" fails.
                        # But we already popped y and added x1. So size is back to original.
                        # We just changed y to x1.
                        # This is a valid local search move.
                        # We don't reset no_improvement_count for a swap, only for a grow.
                        pass 
                    # If found_x1 was False, we reverted nothing, but we removed y.
                    # We must put y back!
                    if not found_x1:
                        current_set.append(y_to_remove)
                        current_sums.update(sums_to_remove)
                        current_size += 1
                        no_improvement_count += 1
                        continue
                        
                    # If found_x1 was True, we have a valid set of size original_size.
                    # We accepted the swap.
                    no_improvement_count += 1 # Swaps don't count as immediate improvement in size
                    update_best(current_set, current_sums, current_size)
                    continue
        
        else:
            # Try simple add
            candidates = np.random.choice(n, 50, replace=False) + 1
            added = False
            for x in candidates:
                if x in current_set:
                    continue
                valid, new_sums = is_valid_addition(x, current_set, current_sums)
                if valid:
                    current_set.append(x)
                    current_sums.update(new_sums)
                    current_size += 1
                    no_improvement_count = 0
                    update_best(current_set, current_sums, current_size)
                    added = True
                    break
            if not added:
                no_improvement_count += 1
                continue
                
    return best_set
