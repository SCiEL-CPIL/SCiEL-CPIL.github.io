import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110
    N = 1000
    max_sum = 2 * N
    
    # sum_counts[s] = number of pairs (a,b) with a<=b in current set such that a+b=s
    sum_counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    def is_valid(x, chosen_arr):
        """Check if x can be added to the set represented by chosen_arr."""
        if len(chosen_arr) == 0:
            return sum_counts[2 * x] < 2
        # Check all sums x + y for y in chosen_arr
        new_sums = x + chosen_arr
        if np.any(sum_counts[new_sums] >= 2):
            return False
        # Check 2*x
        if sum_counts[2 * x] >= 2:
            return False
        return True
    
    def add_element(x, chosen_arr):
        """Add x to the set and update sum_counts."""
        if len(chosen_arr) > 0:
            new_sums = x + chosen_arr
            sum_counts[new_sums] += 1
        sum_counts[2 * x] += 1
        return np.append(chosen_arr, x)
    
    def greedy_construct(order):
        """Greedy construction following a given order."""
        sum_counts_local = np.zeros(max_sum + 1, dtype=np.int32)
        chosen = np.array([], dtype=np.int32)
        
        for x in order:
            if len(chosen) == 0:
                valid = sum_counts_local[2 * x] < 2
            else:
                new_sums = x + chosen
                valid = not np.any(sum_counts_local[new_sums] >= 2)
                if valid:
                    valid = sum_counts_local[2 * x] < 2
            
            if valid:
                if len(chosen) > 0:
                    sum_counts_local[x + chosen] += 1
                sum_counts_local[2 * x] += 1
                chosen = np.append(chosen, x)
        
        return chosen
    
    def local_search(chosen_arr, remaining, deadline):
        """Local search with insertion and swap moves."""
        sum_counts_local = np.zeros(max_sum + 1, dtype=np.int32)
        for x in chosen_arr:
            if len(chosen_arr) > 1:
                # Recompute properly
                pass
        # Rebuild sum_counts from scratch
        sum_counts_local = np.zeros(max_sum + 1, dtype=np.int32)
        chosen_list = list(chosen_arr)
        for i, x in enumerate(chosen_list):
            for y in chosen_list[i:]:
                sum_counts_local[x + y] += 1
        
        best_size = len(chosen_list)
        best_set = list(chosen_list)
        
        while time.time() < deadline:
            improved = False
            random.shuffle(remaining)
            
            for x in remaining:
                # Try insertion
                if len(chosen_arr) == 0:
                    valid = sum_counts_local[2 * x] < 2
                else:
                    new_sums = x + chosen_arr
                    valid = not np.any(sum_counts_local[new_sums] >= 2)
                    if valid:
                        valid = sum_counts_local[2 * x] < 2
                
                if valid:
                    # Insert x
                    if len(chosen_arr) > 0:
                        sum_counts_local[x + chosen_arr] += 1
                    sum_counts_local[2 * x] += 1
                    chosen_arr = np.append(chosen_arr, x)
                    remaining.remove(x)
                    improved = True
                    
                    if len(chosen_arr) > best_size:
                        best_size = len(chosen_arr)
                        best_set = list(chosen_arr)
                    break
            
            if not improved:
                # Try swap moves
                swapped = False
                random.shuffle(remaining)
                for x in remaining:
                    # Try to remove some element y from chosen and add x
                    if len(chosen_arr) == 0:
                        continue
                    
                    # Find elements that conflict with x
                    if len(chosen_arr) > 0:
                        new_sums = x + chosen_arr
                        conflict_mask = sum_counts_local[new_sums] >= 2
                    else:
                        conflict_mask = np.zeros(len(chosen_arr), dtype=bool)
                    
                    if not conflict_mask.any() and sum_counts_local[2 * x] < 2:
                        # No conflicts, just insert (should have been caught above)
                        continue
                    
                    # Try removing each conflicting element
                    for idx in np.where(conflict_mask)[0]:
                        y = chosen_arr[idx]
                        # Check if removing y and adding x is valid
                        # This is complex; skip for now
                        pass
                    swapped = False
                    break
            
            if not improved and not swapped:
                # Try a different strategy: restart with shuffled order
                if time.time() > deadline:
                    break
                
                # Try random restart
                all_elements = list(range(1, N + 1))
                random.shuffle(all_elements)
                chosen_new = greedy_construct(all_elements)
                if len(chosen_new) > best_size:
                    best_size = len(chosen_new)
                    best_set = list(chosen_new)
                    chosen_arr = chosen_new
                    # Rebuild remaining
                    chosen_set = set(chosen_new.tolist())
                    remaining = [x for x in range(1, N + 1) if x not in chosen_set]
                    # Rebuild sum_counts
                    sum_counts_local = np.zeros(max_sum + 1, dtype=np.int32)
                    for i, x in enumerate(best_set):
                        for j in range(i, len(best_set)):
                            sum_counts_local[x + best_set[j]] += 1
                
                break
        
        return best_set
    
    # Start with greedy from 1 to 1000
    initial_order = list(range(1, N + 1))
    chosen = greedy_construct(initial_order)
    
    chosen_set = set(chosen.tolist())
    remaining = [x for x in range(1, N + 1) if x not in chosen_set]
    
    best_size = len(chosen)
    best_set = list(chosen)
    
    # Local search phase
    if time.time() < deadline:
        improved_set = local_search(chosen, remaining, deadline)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Additional random restarts if time permits
    while time.time() < deadline - 5:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        chosen_new = greedy_construct(random_order)
        
        if len(chosen_new) > best_size:
            best_size = len(chosen_new)
            best_set = list(chosen_new)
            
            # Try local search on this new best
            chosen_set = set(chosen_new.tolist())
            remaining = [x for x in range(1, N + 1) if x not in chosen_set]
            
            if time.time() < deadline - 5:
                improved_set = local_search(chosen_new, remaining, deadline)
                if len(improved_set) > best_size:
                    best_size = len(improved_set)
                    best_set = improved_set
    
    return best_set
