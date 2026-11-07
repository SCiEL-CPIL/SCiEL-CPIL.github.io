import random
import itertools
import time

def _get_subset_sums(nums):
    """Generate all non-empty subset sums of nums."""
    n = len(nums)
    sums = set()
    # Bitmask iteration
    for i in range(1, 1 << n):
        s = 0
        for j in range(n):
            if i & (1 << j):
                s += nums[j]
        sums.add(s)
    return sums

def _is_valid(sums):
    """
    Check if the set of subset sums is valid:
    1. No duplicates (handled by set, but we need to ensure the generation didn't miss the 'distinct subsets' rule? 
       The rule says: 'Two different subsets with the same sum still make the set invalid'.
       So we must check if the number of unique sums equals 2^n - 1.
    2. No sum divides another.
    """
    # We pass the set of sums. We also need the count of subsets to check for collisions.
    # Let's modify the check to take nums as well to verify distinctness properly if needed, 
    # or just assume the set of sums is built from distinct subsets?
    # Actually, if two subsets have the same sum, the set `sums` will have size < 2^n - 1.
    # So we need to know the expected count.
    
    # Wait, the helper `_get_subset_sums` returns a set. 
    # If there are collisions, len(sums) < 2^n - 1.
    # So the caller must check len(sums) == (1 << len(nums)) - 1.
    
    # Divisibility check:
    # Sort sums
    sorted_sums = sorted(sums)
    # For each s, check multiples
    # Optimization: If 1 is in sums, invalid (unless len==1).
    if 1 in sums and len(sums) > 1:
        return False
        
    max_sum = sorted_sums[-1] if sorted_sums else 0
    sums_set = sums # for O(1) lookup
    
    for s in sorted_sums:
        if s == 0: continue
        # Check multiples 2s, 3s, ...
        # Only need to check if 2s is in sums? No, 3s could be.
        # But if 2s is not in sums, 3s could be.
        # However, if 2s is not in sums, it doesn't mean 3s isn't.
        # We must check all multiples.
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def _full_validity_check(nums):
    """
    Returns True if nums is a valid set.
    """
    n = len(nums)
    if n == 0:
        return True
        
    sums = _get_subset_sums(nums)
    
    # Check for duplicate sums
    if len(sums) != (1 << n) - 1:
        return False
        
    return _is_valid(sums)

def construct():
    # Start with a known small valid set
    # Try to find a good seed
    # Small primes often work well?
    # Let's start with [2, 3, 5, 7] ?
    # 2, 3 -> sums: 2, 3, 5. No divisibility.
    # Add 5 -> sums: 2,3,5, 2+3=5 (collision!). So [2,3,5] is invalid because 2+3=5 and {5} exists.
    # We need distinct subset sums.
    # Superincreasing sequences guarantee distinct subset sums.
    # e.g., 1, 2, 4, 8... but 1 divides everything. So start with 2?
    # 2, 3, 7, 15... ?
    # 2, 3 -> 2,3,5
    # 2, 3, 7 -> 2,3,5,7, 2+7=9, 3+7=10, 2+3+7=12.
    # Sums: 2,3,5,7,9,10,12.
    # Divisibility: 2 divides 10? Yes. Invalid.
    
    # Let's use a superincreasing sequence starting with a large number to avoid small divisors.
    # Or just use random search.
    
    # Simulated Annealing
    best_set = []
    best_size = 0
    
    # Initialize with a few random small sets
    # Or start empty and add?
    
    # Let's try to build a set by adding elements that don't create conflicts.
    # This is basically the greedy approach but with backtracking?
    
    # Let's use a random local search.
    # State: list of numbers.
    # Score: len(state) if valid, else -infinity.
    
    # We will try to maximize size.
    
    # Start with a valid set of size 3 or 4.
    # [10, 11, 13]
    # Sums: 10, 11, 13, 21, 23, 24, 34.
    # 10 divides 20? No 20. 10 divides 30? No.
    # 11 divides 22? No.
    # 13 divides 26? No.
    # 21 divides 42? No.
    # Looks valid.
    
    current = [10, 11, 13]
    
    if not _full_validity_check(current):
        current = [10, 11] # Fallback
    
    best_set = current[:]
    best_size = len(current)
    
    # Local Search
    # We will try to add elements.
    # If we can't add, we try swaps.
    
    candidates = list(range(1, 257))
    
    # Run for a fixed number of iterations or time
    start_time = time.time()
    time_limit = 50.0 # Leave 10s for overhead
    
    # To make it deterministic-ish but effective, we use a fixed seed for reproducibility in testing, 
    # but the task allows randomness. The evaluator runs it once.
    # We will use random seed based on time? No, standard practice in these tasks is often to allow randomness 
    # but the "seed program" implies determinism might be preferred or at least stable results.
    # However, "random restarts" was in the previous code.
    # Let's use a fixed seed for the random module to ensure we don't get a bad luck run, 
    # but try multiple random restarts.
    
    for restart in range(20):
        random.seed(42 + restart)
        # Start from a random small valid set or the best found so far?
        # Restarting from best found can get stuck in local optima.
        # Restarting from scratch explores more.
        
        # Let's start from a random permutation of candidates and greedily add?
        # That's the previous approach.
        
        # New approach: Hill Climbing on the set.
        # Start with a random valid set of size 3.
        # Pick 3 random distinct numbers. Check validity.
        # If valid, start hill climbing.
        
        current_set = []
        # Find a valid seed of size 3
        for _ in range(100):
            seed_set = random.sample(range(1, 257), 3)
            if _full_validity_check(seed_set):
                current_set = seed_set
                break
        
        if not current_set:
            current_set = [10, 11, 13]
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        # Hill Climb
        # Try to add a random element.
        # If valid, accept.
        # If invalid, try to swap?
        
        # We define a move:
        # 1. Pick a random candidate c not in current_set.
        # 2. Try adding c.
        # 3. If valid, accept.
        # 4. If invalid, pick a random element e in current_set, remove e, add c.
        # 5. If valid, accept (size stays same, but composition changes, might allow future adds).
        # 6. If invalid, reject.
        
        # To escape local optima (where no add is possible), we allow swaps.
        # We also allow "kick" moves: remove 2 elements, add 2 random ones?
        
        # Let's implement a simple loop:
        # While time < limit:
        #   Try to add.
        #   If success, continue.
        #   If fail, try swap.
        #   If success, continue.
        #   If fail, try kick (remove 1, add 2? No, size change. Remove 2, add 1? Size decreases.
        #   We want to maximize size. So we prioritize adds.
        #   If we can't add, we do swaps to change the set.
        
        # We need to track if we've improved recently.
        
        # Optimization: Cache validity? Hard with changing sets.
        
        # Let's just run a fixed number of steps per restart.
        steps = 500
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            if not current_set:
                break
                
            # Try Add
            if len(current_set) < 256:
                candidates_not_in_set = [c for c in candidates if c not in current_set]
                if candidates_not_in_set:
                    c = random.choice(candidates_not_in_set)
                    new_set = current_set + [c]
                    if _full_validity_check(new_set):
                        current_set = new_set
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
                        continue # Accept and try next add
                        
            # Try Swap
            # Remove one, add one
            if len(current_set) >= 1:
                # Pick random element to remove
                idx = random.randint(0, len(current_set) - 1)
                remove_elem = current_set[idx]
                
                # Pick random candidate to add
                candidates_not_in_set = [c for c in candidates if c not in current_set]
                if candidates_not_in_set:
                    add_elem = random.choice(candidates_not_in_set)
                    
                    # Construct new set
                    new_set = current_set[:idx] + current_set[idx+1:] + [add_elem]
                    
                    if _full_validity_check(new_set):
                        current_set = new_set
                        # Size didn't change, so best_size doesn't update unless we add later
                        # But this might enable a future add
                        continue
                        
            # If neither add nor swap worked (or no candidates), try a kick?
            # Or just break this restart if stuck for a while?
            # For simplicity, if we fail add and swap, we just try again.
            # To avoid infinite loops of failing swaps, we can track failures.
            
            # Let's add a simple "kick": if we have failed to add for a while,
            # we might want to remove a random element to reduce constraints?
            # But that reduces size. We only want to do that if it leads to a larger set later.
            # Simulated Annealing would accept a worse state (smaller size) with some probability.
            # Since we are maximizing size, and validity is a hard constraint,
            # we can't accept invalid states.
            # We can accept a valid state with smaller size?
            # Yes, if it allows growth later.
            
            # Let's implement a simple "downhill" move with probability p.
            # If we can't add, and we can't swap, we might try to remove an element.
            # This is risky.
            
            # Let's just rely on Add and Swap.
            
    # Final check and return
    if not _full_validity_check(best_set):
        # Fallback to a known valid small set
        return [10, 11, 13]
        
    return [int(x) for x in best_set]
