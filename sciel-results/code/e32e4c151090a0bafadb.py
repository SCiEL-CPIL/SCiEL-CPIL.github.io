import time
import random

def is_valid_addition(current_sums, x):
    """
    Check if adding x to a set with subset sums 'current_sums' results in a valid set.
    New sums are current_sums union {s + x for s in current_sums} union {x}.
    We need to check all pairs in the new set of sums.
    """
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    
    # Convert to list for pair checking
    sums_list = list(new_sums)
    n = len(sums_list)
    
    for i in range(n):
        a = sums_list[i]
        for j in range(i + 1, n):
            b = sums_list[j]
            if a == b:
                continue
            if a % b == 0 or b % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin
    
    best_set = []
    
    # Try multiple strategies
    strategies = [
        "increasing",
        "decreasing",
        "random"
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
            
        if strategy == "increasing":
            candidates = list(range(2, 257))
        elif strategy == "decreasing":
            candidates = list(range(256, 1, -1))
        else:
            candidates = list(range(2, 257))
            random.shuffle(candidates)
        
        current_set = []
        current_sums = set()
        
        # Greedy add
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Optimization: if x is too small, it might divide existing sums?
            # Just check validity
            if is_valid_addition(current_sums, x):
                current_set.append(x)
                # Update sums
                old_sums = list(current_sums)
                current_sums.update(s + x for s in old_sums)
                current_sums.add(x)
                
                if len(current_set) > len(best_set):
                    best_set = current_set.copy()
        
        # If we found a large set, we might stop early if we hit the theoretical max?
        # Max size is not known, but likely < 10.
        # If we found 6 or more, it's good.
        
    # Final check and return
    if not best_set:
        return [2]
        
    # Verify best_set
    sums = set()
    for i in range(len(best_set)):
        mask = (1 << i) - 1
        # This is inefficient, let's just recompute
    # Recompute sums for verification
    valid_sums = set()
    for mask in range(1, 1 << len(best_set)):
        s = 0
        for i in range(len(best_set)):
            if mask & (1 << i):
                s += best_set[i]
        valid_sums.add(s)
        
    sums_list = list(valid_sums)
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if a % b == 0 or b % a == 0:
                # Invalid, return a known valid small set
                return [3, 5, 23]
                
    return best_set
