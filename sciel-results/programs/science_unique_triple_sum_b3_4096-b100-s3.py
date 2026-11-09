import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leaving 5s margin
    
    max_val = 4096
    candidates = list(range(1, max_val + 1))
    
    best_set = []
    best_size = 0
    
    # Helper to check if candidate x can be added to current_set
    # current_set: list of ints (sorted)
    # sum_set: set of all valid triple sums from current_set
    def can_add(x, current_set, sum_set):
        n = len(current_set)
        
        # Check (x, x, x)
        if 3 * x in sum_set:
            return False, None
        
        # Check (x, x, y) for all y in current_set
        for y in current_set:
            s = 2 * x + y
            if s in sum_set:
                return False, None
        
        # Check (x, y, y) for all y in current_set
        for y in current_set:
            s = x + 2 * y
            if s in sum_set:
                return False, None
        
        # Check (x, y, z) for all pairs y < z in current_set
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                s = x + y + z
                if s in sum_set:
                    return False, None
        
        # All checks passed, compute new sums to add
        new_sums = []
        new_sums.append(3 * x)
        for y in current_set:
            new_sums.append(2 * x + y)
            new_sums.append(x + 2 * y)
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                new_sums.append(x + y + z)
        
        return True, new_sums

    # Run multiple trials with different strategies
    trial = 0
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        # Alternate between random shuffle and sorted approaches
        if trial % 3 == 0:
            random.shuffle(candidates)
        elif trial % 3 == 1:
            candidates_sorted = sorted(candidates, key=lambda x: x % 7)
            random.shuffle(candidates_sorted)
            candidates = candidates_sorted
        else:
            random.shuffle(candidates)
        
        current_set = []
        sum_set = set()
        
        for x in candidates:
            if time.time() > deadline - 3:
                break
            ok, new_sums = can_add(x, current_set, sum_set)
            if ok:
                current_set.append(x)
                sum_set.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # If we hit a good size, we might stop early
            if best_size >= 40:
                break
                
        trial += 1
        
    return best_set
