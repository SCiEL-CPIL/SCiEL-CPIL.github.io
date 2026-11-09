"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(elements):
        """Check if the list 'elements' is sum-free."""
        s = set(elements)
        for a in s:
            if 2 * a in s:
                return False
            for b in s:
                if b != a and (a + b) in s:
                    return False
        return True

    def can_add(s, y):
        """Check if y can be added to set s while keeping it sum-free."""
        if 2 * y in s:
            return False
        for x in s:
            if x == y:
                continue
            # Check if x + y is in s
            if (x + y) in s:
                return False
            # Check if y - x is in s (which means x + (y-x) = y)
            if (y - x) in s:
                return False
        return True

    def greedy_init(elements):
        """Greedy initialization: sort by abs value descending, add if possible."""
        s = set()
        # Sort by absolute value descending to prioritize "large" numbers
        # which are less likely to be sums of two other numbers in the set
        sorted_elems = sorted(elements, key=lambda x: -abs(x))
        for x in sorted_elems:
            if can_add(s, x):
                s.add(x)
        return s

    def odd_init(elements):
        """Initialize with all odd numbers."""
        return set(x for x in elements if x % 2 != 0)

    def high_mag_init(elements):
        """Initialize with numbers having high absolute value."""
        if not elements:
            return set()
        max_abs = max(abs(x) for x in elements)
        threshold = max_abs / 2.0
        return set(x for x in elements if abs(x) > threshold)

    def simulated_annealing(elements, initial_set, time_limit):
        """
        Simulated Annealing to maximize the size of a sum-free subset.
        """
        s = set(initial_set)
        current_size = len(s)
        best_size = current_size
        best_set = set(s)
        
        # Precompute list of elements for random selection
        elems_list = list(elements)
        
        # SA Parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.995 # Cooling rate
        
        # We will run for a fixed number of steps or until time limit
        # Estimate steps: 1 step ~ 100-200us?
        # Let's just loop and check time.
        
        while time.time() - start_time < time_limit:
            # Cooling
            T = max(T * alpha, T_min)
            
            # Generate Move
            move_type = random.random()
            
            if move_type < 0.6:
                # Try to Add
                y = random.choice(elems_list)
                if y not in s:
                    if can_add(s, y):
                        s.add(y)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = set(s)
                    # If invalid, do nothing (reject)
            elif move_type < 0.8:
                # Try to Remove
                # Only remove if we have some size to spare or to escape local optimum
                if s:
                    x = random.choice(list(s))
                    # Probability of accepting removal: exp(-1/T)
                    if random.random() < math.exp(-1.0 / T):
                        s.remove(x)
                        current_size -= 1
            else:
                # Try to Swap
                if s:
                    x = random.choice(list(s))
                    y = random.choice(elems_list)
                    if y not in s:
                        # Tentative swap
                        s.remove(x)
                        if can_add(s, y):
                            s.add(y)
                            # Size unchanged, but structure changed
                            # Accept with prob 1 (neutral move)
                            # Update best if needed? Size is same, so no.
                            # But maybe this leads to better future moves?
                            pass
                        else:
                            # Roll back
                            s.add(x)
                            # Reject swap
                        # Note: We don't track best_size for swaps since size doesn't change
                        # But we might want to track if this swap allows future adds.
                        # The SA process handles this via the state s.
            
            # Periodic restart or reset?
            # If stuck for too long, maybe reset to best_set?
            # For now, rely on cooling.

        return best_set

    def solve_instance(elements):
        # Try multiple initializations
        init1 = greedy_init(elements)
        init2 = odd_init(elements)
        init3 = high_mag_init(elements)
        
        # Pick the largest initial set
        best_init = max([init1, init2, init3], key=len)
        
        # Run SA
        # Allocate time based on instance size?
        # Total time budget 110s.
        # 3 instances. ~35s each.
        # But we should check global time.
        
        # Let's run SA for a fixed number of iterations or time slice.
        # To be safe, we run SA for each instance sequentially, checking time.
        
        # For this instance, we can use the remaining time / 3?
        # Or just run until the global deadline is approached.
        
        # Let's assign a time slice for this instance.
        # If this is the first instance, we have more time.
        # Simple heuristic: use 1/3 of remaining time for this instance?
        # No, just run SA with a tight loop that checks global time.
        
        # To ensure we finish, we will limit the SA loop steps or time.
        # Let's say we allow SA to run until global time is 105s.
        
        result = simulated_annealing(elements, best_init, 105)
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > 105:
            # Not enough time, return empty or best effort
            results.append([])
            continue
            
        elems = instance(size, seed)
        
        # For smaller instances, we can be more exhaustive or use more time.
        # For larger, we need to be efficient.
        
        subset = solve_instance(elems)
        results.append(list(subset))
        
        # Verify sum-free (safety check, should not fail)
        # if not is_sum_free(subset):
        #     # Fallback to greedy
        #     subset = list(greedy_init(elems))
        #     results[-1] = subset

    return results
# EVOLVE-BLOCK-END
