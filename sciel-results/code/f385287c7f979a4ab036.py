import random
import time
import sys

sys.setrecursionlimit(10000)

def construct():
    """
    Returns a list of ints (size 1..30, values 1..256) such that:
    1. All elements are distinct.
    2. All non-empty subset sums are distinct (i.e., no two different subsets have the same sum).
    3. No subset sum divides another.
    Maximize the number of elements.
    """
    
    def _check_validity(nums):
        n = len(nums)
        if n == 0:
            return True
        if n > 12:
            # Too expensive to check all subsets for n > 12 in a tight loop
            # But we can do a quick probabilistic check or just return False to prune
            # Actually, for the local search, we might have larger sets.
            # Let's use a faster check for larger n or skip if n is too big for full check?
            # The problem says "Maximize set size". If we reach size 13+, we need to check.
            # 2^13 = 8192 subsets. Checking pairs is 8192^2 / 2 ~ 33M. Too slow in Python.
            # We need an efficient check.
            
            # Efficient check for divisibility:
            # Generate all sums. If len(sums) < 2^n - 1, invalid.
            # Then check if any s in sums divides another s' in sums.
            # This requires O(M^2) where M is number of sums.
            # For n=13, M=8191. 8191^2 is ~67M. Still too slow for a local search step.
            
            # We need to restrict the search space or use a faster heuristic.
            # Or, we can only check validity for n <= 10 in the local search steps,
            # and rely on the final verification to be more robust?
            # No, if we accept an invalid state, we can't grow it.
            
            # Alternative: Use a randomized check for large n?
            # Or, realize that superincreasing sequences are likely the only way to get large n
            # with distinct sums and no divisibility.
            
            # Let's implement the check with an early exit and optimized loops.
            
            sums = set()
            # Generate sums
            for mask in range(1, 1 << n):
                s = 0
                m = mask
                idx = 0
                while m:
                    if m & 1:
                        s += nums[idx]
                    m >>= 1
                    idx += 1
                # Quick check: if s is in sums, we have a collision? 
                # No, we need to check divisibility.
                # But if s is already in sums, that's a duplicate sum, which is invalid.
                if s in sums:
                    return False
                sums.add(s)
            
            # Check divisibility
            # Sort sums for faster checking?
            # For each s in sums, check if 2s, 3s, ... are in sums.
            # To optimize, we can iterate over sorted sums.
            sorted_sums = sorted(sums)
            sums_set = sums
            max_sum = sorted_sums[-1]
            
            for s in sorted_sums:
                # Check multiples
                # If s is small, there are many multiples.
                # If s is large, few multiples.
                # We can break early if 2*s > max_sum.
                if 2 * s > max_sum:
                    continue
                multiple = 2 * s
                while multiple <= max_sum:
                    if multiple in sums_set:
                        return False
                    multiple += s
            return True

    def _fast_check_small(nums):
        """
        Optimized check for n <= 10.
        """
        n = len(nums)
        if n == 0:
            return True
        sums = []
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += nums[i]
            sums.append(s)
        
        # Check duplicates
        if len(set(sums)) != len(sums):
            return False
            
        sums_set = set(sums)
        max_sum = max(sums)
        
        for s in sums:
            if 2 * s > max_sum:
                continue
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in sums_set:
                    return False
                multiple += s
        return True

    # Strategy:
    # 1. Find a valid seed set of size 3-5.
    # 2. Use local search to expand.
    #    - Moves: Add a random number, Swap a number, (Remove if stuck? No, we want to grow).
    #    - We prioritize Add. If Add fails, we do Swaps to find a better configuration.
    #    - To avoid the expensive check for n > 10, we can limit the local search to n <= 10
    #      and then try to extend from the best n=10 found?
    #      Or, we can just use the slower check if n > 10, but fewer iterations.
    
    time_limit = 55.0
    start_time = time.time()
    
    best_set = []
    best_size = 0
    best_sums = None # Cache sums? No, set changes.
    
    # Predefined good starting points
    # Let's try to find a size 3 seed quickly
    seed_candidates = [
        [10, 11, 13],
        [11, 13, 17],
        [13, 17, 19],
        [19, 23, 29]
    ]
    
    for seed in seed_candidates:
        if _fast_check_small(seed):
            current_set = seed[:]
            break
    else:
        # Random seed
        while True:
            seed = random.sample(range(10, 257), 3)
            if _fast_check_small(seed):
                current_set = seed
                break

    best_set = current_set[:]
    best_size = len(current_set)
    
    # Local Search
    # We will run multiple restarts.
    # Each restart starts from a random valid small set or the best found.
    # We will use a time-based loop.
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize current set
        if restart == 0:
            current_set = best_set[:]
        else:
            # Start from a random valid seed
            found_seed = False
            for _ in range(20):
                seed = random.sample(range(10, 257), 3)
                if _fast_check_small(seed):
                    current_set = seed
                    found_seed = True
                    break
            if not found_seed:
                current_set = [10, 11, 13]
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        stuck_count = 0
        max_stuck = 100
        
        while time.time() - start_time < time_limit:
            if len(current_set) >= 30:
                break
                
            # 1. Try to Add
            added = False
            # Sample a few random candidates to try adding
            # Instead of trying all 256, try a random subset
            # We want to add numbers that are "large" relative to the sum?
            # Large numbers are less likely to divide or be divided by?
            # Actually, if s is large, 2s is likely > max_sum, so no divisibility issue.
            # But s itself might divide some other sum?
            # If we add a large number L, the new sums are old_sums U (old_sums + L) U {L}.
            # New divisibility risks:
            # a) L divides old_s? Unlikely if L is large.
            # b) old_s divides L? Possible.
            # c) old_s divides (old_t + L)?
            # d) (old_s + L) divides (old_t + L)?
            
            # Let's just pick random candidates.
            candidates = [c for c in range(1, 257) if c not in current_set]
            random.shuffle(candidates)
            
            tried = 0
            for c in candidates:
                if tried > 50:
                    break
                tried += 1
                new_set = current_set + [c]
                n = len(new_set)
                if n <= 10:
                    valid = _fast_check_small(new_set)
                else:
                    valid = _check_validity(new_set)
                    
                if valid:
                    current_set = new_set
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                    added = True
                    stuck_count = 0
                    break
            
            if added:
                continue
                
            # 2. Try Swap
            swapped = False
            if len(current_set) > 0:
                idx = random.randint(0, len(current_set) - 1)
                remove_elem = current_set[idx]
                
                # Try a few random swaps
                candidates = [c for c in range(1, 257) if c not in current_set]
                random.shuffle(candidates)
                tried = 0
                for c in candidates:
                    if tried > 50:
                        break
                    tried += 1
                    new_set = current_set[:idx] + current_set[idx+1:] + [c]
                    n = len(new_set)
                    if n <= 10:
                        valid = _fast_check_small(new_set)
                    else:
                        valid = _check_validity(new_set)
                    
                    if valid:
                        current_set = new_set
                        swapped = True
                        stuck_count = 0
                        break
                        
            if swapped:
                continue
                
            # 3. Stuck
            stuck_count += 1
            if stuck_count > max_stuck:
                break
        
    # Final verification of best_set
    if best_size > 10:
        if not _check_validity(best_set):
            # Fallback
            best_set = [10, 11, 13]
            best_size = 3
    else:
        if not _fast_check_small(best_set):
            best_set = [10, 11, 13]
            best_size = 3
            
    # Ensure types are correct
    return [int(x) for x in best_set]
