import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit, 5s margin
    
    # Initial solution: Powers of 2
    # This is a valid solution with max 2048
    current_set = [1 << i for i in range(12)]
    best_max = max(current_set)
    best_set = current_set[:]
    
    def is_valid(s):
        sums = set()
        # Generate all subset sums
        # n=12, 4096 sums
        # Use bit manipulation or iterative addition
        # Iterative is fast enough
        curr_sums = {0}
        for x in s:
            new_sums = set()
            for val in curr_sums:
                new_sums.add(val + x)
            curr_sums.update(new_sums)
            if len(curr_sums) != 2 ** (s.index(x) + 1): # Optimization: check size early?
                # Actually, just check at the end or periodically
                pass
        return len(curr_sums) == 4096

    # Better validity check
    def check_validity(s):
        sums = {0}
        for x in s:
            new_sums = {val + x for val in sums}
            sums.update(new_sums)
        return len(sums) == 4096

    # Local Search
    # We want to minimize max(s)
    # We will try to reduce the largest element
    
    # To make search faster, we can try to "compress" the set
    # Strategy: Randomly perturb the set, then try to reduce the max
    
    while time.time() < deadline:
        # Pick a random element to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Try to change it to a smaller value
        # Or random value in a range
        # If idx is the largest, try to reduce it
        # If not, randomize to allow restructuring
        
        # Let's try to reduce the current max
        max_idx = current_set.index(max(current_set))
        
        # Try to reduce the max element
        if random.random() < 0.5:
            target_idx = max_idx
        else:
            target_idx = random.randint(0, 11)
            
        # Try a random new value smaller than the current one
        # Or just random in [1, best_max]
        new_val = random.randint(1, best_max)
        
        # Ensure distinctness
        if new_val in current_set:
            continue
            
        current_set[target_idx] = new_val
        current_set.sort()
        
        if check_validity(current_set):
            curr_max = max(current_set)
            if curr_max < best_max:
                best_max = curr_max
                best_set = current_set[:]
            # Accept if same max to allow exploration?
            # Yes, but keep best
        else:
            # Revert
            current_set[target_idx] = old_val
            current_set.sort()
            
    return best_set
