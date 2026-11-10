import random
import time

def check_valid(a):
    """Check if all subset sums are distinct."""
    n = len(a)
    sums = [0]
    for x in a:
        # Add x to all existing sums
        sums += [s + x for s in sums]
    # Check for duplicates
    if len(set(sums)) != len(sums):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Leave 5s margin for 60s limit
    
    # Initial guess: Powers of 2
    best_a = [1 << i for i in range(12)]
    best_max = max(best_a)
    
    # Randomized Local Search
    # We try to reduce the maximum element.
    # Strategy: Start from powers of 2.
    # Randomly pick an index, decrease its value, check validity.
    # If valid, keep it.
    # If invalid, revert.
    # To escape local minima, we can also increase values slightly or swap.
    
    current_a = best_a[:]
    current_max = best_max
    
    # Pre-compute subset sums for faster checking?
    # For n=12, 4096 sums. Recalculating is ~4096 ops. Very fast.
    # Python loop might be slow, but 4096 is small.
    # Let's optimize check_valid using numpy if possible, or just set.
    
    def is_valid(a):
        sums = {0}
        for x in a:
            new_sums = {s + x for s in sums}
            # If any sum in new_sums is already in sums (excluding 0 which is handled), it's a collision?
            # No, we are building the set.
            # If we just union, we lose the count.
            # We need to check if the size of the union is 2 * previous size.
            if len(sums | new_sums) != len(sums) + len(new_sums):
                return False
            sums |= new_sums
        return True

    # Update is_valid to be faster
    def is_valid_fast(a):
        sums = [0]
        for x in a:
            sums += [s + x for s in sums]
        return len(set(sums)) == 4096

    # Use is_valid_fast
    # Actually, set construction is O(N). List concatenation is O(N).
    # For N=4096, it's fine.
    
    random.seed(42)
    
    # To improve search, we can use a "cooling" factor or just random restarts.
    # Given the small dimension, random restarts from powers of 2 with random perturbations might work.
    
    # A better heuristic:
    # The elements should be roughly sorted.
    # Let's try to minimize the largest element by targeting it directly.
    
    # We will use a simple hill climbing with random restarts.
    # Since n=12 is small, we can also try a systematic reduction of the largest element.
    
    # Let's try a different approach: 
    # Start with a known good sequence if we can find one, or generate via greedy with backtracking.
    
    # Given the time, let's stick to Randomized Local Search (RLS).
    # State: list of 12 ints.
    # Move: pick random i, change a[i] to a[i] + delta (delta in [-10, 10]).
    # Constraint: a[i] >= 1, distinct elements.
    # Objective: minimize max(a).
    
    # To make it effective, we should bias towards reducing large elements.
    
    for _ in range(1000000):
        if time.time() > deadline:
            break
            
        # Pick an index to modify
        # Bias towards larger indices (larger values)
        i = random.randint(0, 11)
        
        # Try to decrease
        if current_a[i] > 1:
            # Random decrease
            dec = random.randint(1, 5)
            new_val = current_a[i] - dec
            if new_val < 1:
                new_val = 1
            
            # Ensure distinctness
            if new_val in current_a[:i] or new_val in current_a[i+1:]:
                continue
                
            old_val = current_a[i]
            current_a[i] = new_val
            
            if is_valid_fast(current_a):
                new_max = max(current_a)
                if new_max < best_max:
                    best_max = new_max
                    best_a = current_a[:]
            else:
                current_a[i] = old_val
        else:
            # If it's 1, maybe increase or swap?
            # Or just skip.
            pass

    return best_a
