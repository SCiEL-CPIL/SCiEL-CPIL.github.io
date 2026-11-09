import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    max_val = 4096
    max_sum = 3 * max_val
    
    # Use a list of booleans for sum existence for O(1) fast lookup
    # Size is max_sum + 1
    sum_exists = [False] * (max_sum + 1)
    
    best_set = []
    best_size = 0
    
    # Helper to add a number to the current set
    # current_set: list of ints
    # Returns True if added, False if conflict
    def try_add(x, current_set):
        # Check all triples (x, y, z) where y, z <= x (in value, but we process in order)
        # Actually, since we add in increasing order of 'x' (or some order), 
        # we only need to check triples where the new element is the largest or equal.
        # Wait, the order of addition doesn't have to be sorted by value.
        # But if we pick a candidate x, we must check all triples involving x and any two elements already in the set.
        # Let S be the current set.
        # Triples to check: (x, x, x), (x, x, y), (x, y, y) for y in S, and (x, y, z) for y, z in S, y <= z.
        
        n = len(current_set)
        
        # 1. (x, x, x)
        s = 3 * x
        if sum_exists[s]:
            return False
        
        # 2. (x, x, y) for y in current_set
        for y in current_set:
            s = 2 * x + y
            if sum_exists[s]:
                return False
        
        # 3. (x, y, y) for y in current_set
        for y in current_set:
            s = x + 2 * y
            if sum_exists[s]:
                return False
        
        # 4. (x, y, z) for y < z in current_set
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                s = x + y + z
                if sum_exists[s]:
                    return False
        
        # If all checks pass, add x and update sum_exists
        sum_exists[3 * x] = True
        for y in current_set:
            sum_exists[2 * x + y] = True
            sum_exists[x + 2 * y] = True
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                sum_exists[x + y + z] = True
        
        return True

    # Strategy: Try multiple different orderings of candidates.
    # 1. Standard ascending
    # 2. Ascending with small perturbations
    # 3. Random shuffles (but few, as they are slow)
    # 4. Modular bias orders
    
    candidates = list(range(1, max_val + 1))
    
    # Pre-define some ordering strategies
    # Strategy 0: Ascending
    # Strategy 1: Descending (might be bad, but let's see) - Actually, ascending is usually better for greedy packing.
    # Let's try ascending with different "steps" or biases.
    
    orders = []
    
    # 1. Pure ascending
    orders.append(candidates[:])
    
    # 2. Ascending, but skip some small numbers? No, greedy usually takes small numbers.
    # Instead, let's try random shuffles but limit the number of trials.
    
    # We will run trials. Each trial resets sum_exists and current_set.
    # To speed up, we can pre-generate a few random orders.
    
    num_random_trials = 10
    random_orders = []
    for _ in range(num_random_trials):
        c = candidates[:]
        random.shuffle(c)
        random_orders.append(c)
    
    # Add modular orders
    # Sort by x % 3, then x
    c_mod3 = sorted(candidates, key=lambda x: (x % 3, x))
    orders.append(c_mod3)
    
    # Sort by x % 5, then x
    c_mod5 = sorted(candidates, key=lambda x: (x % 5, x))
    orders.append(c_mod5)
    
    # Sort by x % 7, then x
    c_mod7 = sorted(candidates, key=lambda x: (x % 7, x))
    orders.append(c_mod7)
    
    # Combine orders
    all_orders = orders + random_orders
    
    for order in all_orders:
        if time.time() > deadline - 2.0:
            break
            
        current_set = []
        # Reset sum_exists
        # Instead of re-allocating, we can track used sums and clear them, 
        # or just allocate a new list. Allocation of 12k bools is fast.
        sum_exists = [False] * (max_sum + 1)
        
        for x in order:
            if time.time() > deadline - 2.0:
                break
                
            if try_add(x, current_set):
                current_set.append(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we reach a very high size, we might stop early?
            # Theoretical max is ~42. If we get 40+, it's very good.
            if best_size >= 42:
                break

    return best_set
