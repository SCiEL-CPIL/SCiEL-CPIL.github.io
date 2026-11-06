"""
Sidon set construction using randomized local search with incremental sum maintenance.
"""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Prescribed elements
    base = [1, 2, 5, 11]
    prescribed_set = set(base)
    
    # Precompute initial sums for base
    def compute_sums(elements):
        sums = set()
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                sums.add(elements[i] + elements[j])
        return sums
        
    initial_sums = compute_sums(base)
    
    best_solution = list(base)
    best_size = len(base)
    best_sums = initial_sums
    
    # Helper to check if adding x to a set with given sums is valid
    # Returns the new sums if valid, None otherwise
    def try_add(x, current_set, current_sums):
        # Check if x is already in set
        if x in current_set:
            return None
        
        new_sums = {x + y for y in current_set} | {2 * x}
        
        # Check for conflicts
        # The number of new sums should be len(current_set) + 1
        # If any sum in new_sums is already in current_sums, it's a conflict
        if len(new_sums) != len(current_set) + 1:
            return None
        if new_sums & current_sums:
            return None
            
        return new_sums

    # Helper to remove an element from the set and update sums
    def remove_element(x, current_set, current_sums):
        if x not in current_set:
            return None, None
            
        remaining_set = current_set - {x}
        
        # Recalculate sums for the remaining set
        # This is O(n^2) but n is small (max 20), so it's fast enough
        new_sums = set()
        sl = sorted(remaining_set)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                new_sums.add(sl[i] + sl[j])
                
        return remaining_set, new_sums

    def local_search(start_set, start_sums, time_limit=10.0):
        nonlocal best_solution, best_size, best_sums
        
        current_set = set(start_set)
        current_sums = set(start_sums)
        current_size = len(current_set)
        
        rng = random.Random()
        
        # Precompute all possible candidates
        all_candidates = list(range(1, 285))
        
        # Initial greedy fill to get a good starting point for local search
        # Try to add elements in a random order
        candidates_to_try = [x for x in all_candidates if x not in current_set]
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            if time.time() > start_time + time_limit:
                break
            new_sums = try_add(x, current_set, current_sums)
            if new_sums is not None:
                current_set.add(x)
                current_sums = new_sums
                current_size += 1
        
        # Update global best
        if current_size > best_size:
            best_size = current_size
            best_solution = list(current_set)
            best_sums = set(current_sums)
            
        # Local search loop
        start_ls = time.time()
        no_improvement_count = 0
        max_no_improvement = 500
        
        while time.time() - start_ls < time_limit and time.time() < deadline - 5.0:
            
            improved = False
            
            # Try to add a random element
            if current_size < 284:
                # Pick a few random candidates to try adding
                for _ in range(10):
                    x = rng.choice(all_candidates)
                    if x in current_set:
                        continue
                    new_sums = try_add(x, current_set, current_sums)
                    if new_sums is not None:
                        current_set.add(x)
                        current_sums = new_sums
                        current_size += 1
                        improved = True
                        break
            
            # Try to swap: remove one non-prescribed, add one new
            removable = [x for x in current_set if x not in prescribed_set]
            if removable and current_size > len(prescribed_set):
                x_remove = rng.choice(removable)
                remaining_set, reduced_sums = remove_element(x_remove, current_set, current_sums)
                
                if remaining_set is not None:
                    # Try to find a valid x_add for the reduced set
                    for _ in range(20):
                        x_add = rng.choice(all_candidates)
                        if x_add in current_set:
                            continue
                        # Check if x_add can be added to remaining_set
                        new_sums_test = {x_add + y for y in remaining_set} | {2 * x_add}
                        
                        # Check validity for remaining_set
                        if len(new_sums_test) == len(remaining_set) + 1 and not (new_sums_test & reduced_sums):
                            # Valid swap
                            current_set = remaining_set | {x_add}
                            current_sums = reduced_sums | new_sums_test
                            # current_size remains the same
                            improved = True
                            break
            
            if improved:
                no_improvement_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_solution = list(current_set)
                    best_sums = set(current_sums)
            else:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    break
        
        return best_size

    # Run multiple restarts with different time allocations
    # First, do a quick greedy to establish a baseline
    current_set = set(base)
    current_sums = set(initial_sums)
    rng = random.Random(42)
    
    # Greedy fill with random order
    candidates = [x for x in range(1, 285) if x not in current_set]
    rng.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline - 20.0:
            break
        new_sums = try_add(x, current_set, current_sums)
        if new_sums is not None:
            current_set.add(x)
            current_sums = new_sums
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_solution = list(current_set)
        best_sums = set(current_sums)
        
    # Now run local search with restarts
    remaining_time = deadline - 5.0 - time.time()
    num_restarts = 10
    time_per_restart = remaining_time / num_restarts if num_restarts > 0 else 0
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Create a new random starting set
        start_set = set(base)
        start_sums = set(initial_sums)
        
        # Random greedy fill
        rng = random.Random(i * 1000 + 42)
        candidates = [x for x in range(1, 285) if x not in start_set]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5.0 - (time_per_restart / 2):
                break
            new_sums = try_add(x, start_set, start_sums)
            if new_sums is not None:
                start_set.add(x)
                start_sums = new_sums
                
        # Run local search on this starting set
        local_search(start_set, start_sums, time_limit=time_per_restart / 2)
        
        # If we reached size 20, we can stop early as it's the maximum possible
        if best_size >= 20:
            break
            
    return best_solution

