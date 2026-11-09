def construct():
    """Construct a Sidon set (distinct pairwise sums) in [1, 284] containing {1, 2, 5, 11}.
    
    Uses Simulated Annealing to maximize the size of the set.
    """
    import random
    import time

    N = 284
    required = [1, 2, 5, 11]
    
    # Use a list to track used sums for O(1) lookup
    # Max sum is 284 + 284 = 568. Min sum 2.
    # We can use a set or a boolean array. Set is fine for N~20.
    
    def get_initial_set():
        """Greedy construction to get a valid starting set."""
        s = sorted(required)
        used_sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                used_sums.add(s[i] + s[j])
                
        candidates = [x for x in range(12, N + 1) if x not in s]
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            new_sums = set()
            for y in s:
                sum_val = x + y
                if sum_val in used_sums:
                    valid = False
                    break
                new_sums.add(sum_val)
            if valid:
                sum_2x = 2 * x
                if sum_2x in used_sums or sum_2x in new_sums:
                    valid = False
                else:
                    new_sums.add(sum_2x)
                    
            if valid:
                s.append(x)
                s.sort()
                used_sums.update(new_sums)
                
        return s, used_sums

    def is_valid_set(s):
        """Check if a set is a valid Sidon set."""
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sum_val = s[i] + s[j]
                if sum_val in sums:
                    return False
                sums.add(sum_val)
        return True

    def calculate_score(s):
        """
        Score = Size.
        We want to maximize size.
        """
        return len(s)

    # Simulated Annealing
    def simulate_annealing(initial_set, initial_sums, steps=50000, start_temp=1.0, cool_rate=0.9995):
        current_set = sorted(initial_set)
        current_sums = set(initial_sums)
        best_set = current_set[:]
        best_size = len(current_set)
        
        temp = start_temp
        
        # Precompute all possible candidates
        all_candidates = list(range(1, N + 1))
        
        for step in range(steps):
            # 1. Try to add a random element
            # 2. Try to swap a random element with a random candidate
            # 3. Try to remove a random element (rarely useful unless to enable swap)
            
            move_type = random.random()
            
            new_set = current_set[:]
            new_sums = set(current_sums)
            
            if move_type < 0.5:
                # Add
                # Pick a random candidate not in set
                # To make it efficient, we can pick from a list of "free" candidates, 
                # but maintaining that list is hard. 
                # Just pick random 1..284.
                x = random.randint(1, N)
                if x in current_set:
                    continue
                    
                # Check if x can be added
                valid = True
                new_sums_add = set()
                for y in current_set:
                    sum_val = x + y
                    if sum_val in new_sums:
                        valid = False
                        break
                    new_sums_add.add(sum_val)
                if valid:
                    sum_2x = 2 * x
                    if sum_2x in new_sums or sum_2x in new_sums_add:
                        valid = False
                    else:
                        new_sums_add.add(sum_2x)
                        
                if valid:
                    new_set.append(x)
                    new_set.sort()
                    new_sums.update(new_sums_add)
                else:
                    continue # Skip if invalid add? Or accept with penalty? 
                    # For max size, we only accept valid adds.
                    # If we can't add, we don't change state.
                    # This makes the search stuck if we can't add.
                    # Better: If add fails, try a swap instead in this iteration.
                    # Let's just restart the iteration logic.
                    continue

            elif move_type < 0.9:
                # Swap
                if not current_set:
                    continue
                # Pick a random element from current_set (must not be required)
                # Required elements are 1, 2, 5, 11
                swappable = [x for x in current_set if x not in required]
                if not swappable:
                    continue
                    
                old_x = random.choice(swappable)
                new_x = random.randint(1, N)
                
                if new_x in current_set or new_x == old_x or new_x in required:
                    continue
                    
                # Remove old_x, add new_x
                # Calculate new sums
                # Sums involving old_x: old_x + y for y in current_set (y != old_x), and 2*old_x
                # Sums involving new_x: new_x + y for y in current_set (y != old_x), and 2*new_x
                
                # Efficient way:
                # 1. Remove sums involving old_x from new_sums
                for y in current_set:
                    if y == old_x:
                        continue
                    sum_val = old_x + y
                    if sum_val in new_sums:
                        new_sums.remove(sum_val)
                # Remove 2*old_x
                sum_2old = 2 * old_x
                if sum_2old in new_sums:
                    new_sums.remove(sum_2old)
                
                # 2. Check if new_x can be added to (current_set - {old_x})
                valid = True
                new_sums_add = set()
                for y in current_set:
                    if y == old_x:
                        continue
                    sum_val = new_x + y
                    if sum_val in new_sums:
                        valid = False
                        break
                    new_sums_add.add(sum_val)
                if valid:
                    sum_2new = 2 * new_x
                    if sum_2new in new_sums or sum_2new in new_sums_add:
                        valid = False
                    else:
                        new_sums_add.add(sum_2new)
                        
                if valid:
                    # Apply swap
                    new_set.remove(old_x)
                    new_set.append(new_x)
                    new_set.sort()
                    new_sums.update(new_sums_add)
                else:
                    continue

            else:
                # Remove (to escape local minima? Usually not needed for max size, 
                # but can help if we are stuck at a local max size and need to restructure)
                # Only remove non-required
                removable = [x for x in current_set if x not in required]
                if not removable:
                    continue
                rem_x = random.choice(removable)
                
                # Remove sums involving rem_x
                for y in current_set:
                    if y == rem_x:
                        continue
                    sum_val = rem_x + y
                    if sum_val in new_sums:
                        new_sums.remove(sum_val)
                sum_2rem = 2 * rem_x
                if sum_2rem in new_sums:
                    new_sums.remove(sum_2rem)
                    
                new_set.remove(rem_x)
                new_set.sort()

            # Evaluate
            current_size = len(current_set)
            new_size = len(new_set)
            
            # Objective: Maximize size.
            # Delta = new_size - current_size
            delta = new_size - current_size
            
            if delta > 0:
                # Always accept
                current_set = new_set
                current_sums = new_sums
                if new_size > best_size:
                    best_set = new_set[:]
                    best_size = new_size
            else:
                # delta <= 0
                # Accept with probability exp(-(-delta) / temp) ? 
                # Since we want to maximize, delta is negative or 0.
                # We want to accept bad moves (delta < 0) to escape local optima.
                # Probability = exp(delta / temp) where delta is negative.
                # Wait, standard SA minimizes cost. Here we maximize value.
                # Cost = -Value. Delta Cost = -Delta Value.
                # If Delta Value < 0, Delta Cost > 0.
                # Accept if random() < exp(-Delta Cost / temp) = exp(Delta Value / temp).
                if random.random() < (temp ** delta): # exp(delta/temp) approx? No.
                    # Correct formula: exp(delta / temp)
                    # If delta is -1 and temp is 1, exp(-1) = 0.36
                    pass
                # Let's use math.exp
                import math
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_set = new_set
                    current_sums = new_sums

            temp *= cool_rate
            if temp < 0.01:
                break

        return best_set

    # Run multiple restarts
    best_global = []
    
    # Initial greedy to get a good start
    init_set, init_sums = get_initial_set()
    
    # Run SA
    result = simulate_annealing(init_set, init_sums, steps=20000, start_temp=1.5, cool_rate=0.999)
    
    if len(result) > len(best_global):
        best_global = result

    # Try another restart with different seed
    random.seed(43)
    init_set2, init_sums2 = get_initial_set()
    result2 = simulate_annealing(init_set2, init_sums2, steps=20000, start_temp=1.5, cool_rate=0.999)
    
    if len(result2) > len(best_global):
        best_global = result2

    # Verify result
    if not all(x in best_global for x in required):
        return required
        
    # Final sanity check
    if not is_valid_set(best_global):
        # Fallback to greedy if SA produced invalid (shouldn't happen if logic is correct)
        return sorted(required + [x for x in range(12, N+1) if can_add_greedy(sorted(required), x)])

    return sorted(best_global)
    
    def can_add_greedy(arr, x):
        used = set()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                used.add(arr[i] + arr[j])
        for y in arr:
            if x + y in used:
                return False
        if 2 * x in used:
            return False
        return True
