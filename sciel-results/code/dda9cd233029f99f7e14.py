import time
import random
import sys

sys.setrecursionlimit(10000)

def _subset_sums(elements):
    """
    Returns a list of all unique non-empty subset sums.
    """
    n = len(elements)
    if n == 0:
        return []
    
    # Use a set to track unique sums
    current_sums = {0}
    for x in elements:
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        current_sums.update(new_sums)
        
    current_sums.discard(0)
    return list(current_sums)

def _is_valid(elements):
    """
    Checks if the set of elements satisfies:
    1. All non-empty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    """
    k = len(elements)
    if k == 0:
        return True
        
    sums = _subset_sums(elements)
    
    # Check for distinct subset sums
    expected_unique = (1 << k) - 1
    if len(sums) != expected_unique:
        return False
        
    # Check for 1
    if 1 in sums and k > 1:
        return False
        
    sums.sort()
    max_sum = sums[-1]
    n = len(sums)
    
    # Check divisibility
    # Only sums <= max_sum / 2 can divide another sum
    for i in range(n):
        a = sums[i]
        if a > max_sum // 2:
            break
            
        for j in range(i + 1, n):
            b = sums[j]
            if b % a == 0:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # 90s limit, keep 5s margin
    
    best_set = []
    best_size = 0
    
    # Candidates 1..256
    candidates = list(range(256, 0, -1))
    
    # Helper to update best
    def update_best(s):
        nonlocal best_set, best_size
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
    # 1. Greedy Descending
    s = []
    for x in candidates:
        if time.time() > deadline:
            break
        if _is_valid(s + [x]):
            s.append(x)
    update_best(s)
    
    # 2. Local Search / Hill Climbing
    in_set = set(best_set)
    not_in_set = [x for x in candidates if x not in in_set]
    
    no_improvement_count = 0
    while time.time() < deadline - 5 and no_improvement_count < 50:
        improved = False
        
        # Try adding a random element
        if not_in_set:
            random.shuffle(not_in_set)
            for x in not_in_set:
                if time.time() > deadline:
                    break
                new_set = best_set + [x]
                if _is_valid(new_set):
                    best_set = new_set
                    in_set.add(x)
                    not_in_set.remove(x)
                    best_size += 1
                    improved = True
                    break
        
        if improved:
            no_improvement_count = 0
            continue
        
        # Try swapping
        if best_set:
            idx = random.randint(0, len(best_set) - 1)
            old_val = best_set[idx]
            # Try to find a replacement y > old_val
            # Filter candidates > old_val not in set
            candidates_to_try = [y for y in candidates if y > old_val and y not in in_set]
            random.shuffle(candidates_to_try)
            
            for y in candidates_to_try:
                if time.time() > deadline:
                    break
                new_set = best_set.copy()
                new_set[idx] = y
                if _is_valid(new_set):
                    in_set.remove(old_val)
                    in_set.add(y)
                    not_in_set.remove(y)
                    not_in_set.append(old_val)
                    best_set = new_set
                    improved = True
                    break
        
        if not improved:
            no_improvement_count += 1
            
    # 3. Random Restarts
    restarts = 0
    while time.time() < deadline - 2 and restarts < 10:
        random_candidates = candidates[:]
        random.shuffle(random_candidates)
        s = []
        for x in random_candidates:
            if time.time() > deadline - 1:
                break
            if _is_valid(s + [x]):
                s.append(x)
        update_best(s)
        restarts += 1

    return best_set
