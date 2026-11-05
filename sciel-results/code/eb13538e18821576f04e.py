import time
import random

def has_4ap_with_x(S, x):
    """
    Checks if adding x to set S creates a 4-term arithmetic progression.
    S is a set of integers.
    x is an integer not in S.
    """
    # x is the 1st term: x, x+d, x+2d, x+3d
    max_d = (512 - x) // 3
    for d in range(1, max_d + 1):
        if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
            return True
    
    # x is the 2nd term: x-d, x, x+d, x+2d
    max_d = min(x - 1, (512 - x) // 2)
    for d in range(1, max_d + 1):
        if (x - d in S) and (x + d in S) and (x + 2*d in S):
            return True
    
    # x is the 3rd term: x-2d, x-d, x, x+d
    max_d = (x - 1) // 2
    for d in range(1, max_d + 1):
        if (x - 2*d in S) and (x - d in S) and (x + d in S):
            return True
    
    # x is the 4th term: x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
            return True
    
    return False

def greedy_fill(S, pool):
    """
    Attempts to add elements from pool to S if they don't create a 4-AP.
    pool is a list of integers.
    Returns the updated set S.
    """
    # Shuffle pool to introduce randomness
    pool = list(pool)
    random.shuffle(pool)
    
    for x in pool:
        if x not in S:
            if not has_4ap_with_x(S, x):
                S.add(x)
    return S

def construct():
    N = 512
    all_nums = list(range(1, N + 1))
    
    # Initial greedy fill
    best_set = set()
    best_set = greedy_fill(best_set, all_nums)
    best_size = len(best_set)
    
    # Keep a few good sets for diversity in kicks
    good_sets = [best_set]
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    while time.time() < deadline:
        # Pick a base set to kick
        base_set = random.choice(good_sets)
        
        # Decide how many elements to remove
        # Removing 1-3 elements allows for significant structural change
        k = random.randint(1, 3)
        
        # Ensure we don't remove more than available
        if len(base_set) <= k:
            # If set is too small, just restart from scratch?
            # Or remove all and refill?
            # For now, if small, just do a full restart
            new_base = set()
            pool = all_nums
        else:
            to_remove = random.sample(list(base_set), k)
            new_base = base_set.copy()
            for x in to_remove:
                new_base.remove(x)
            # Pool is all numbers not in new_base
            # This includes the removed ones and those never added
            pool = [x for x in all_nums if x not in new_base]
        
        # Greedy fill
        new_set = greedy_fill(new_base, pool)
        new_size = len(new_set)
        
        if new_size > best_size:
            best_set = new_set
            best_size = new_size
            # Reset good_sets to focus on the new best
            good_sets = [new_set]
        else:
            # If new_size is close, keep it for diversity
            if new_size >= best_size - 2:
                if len(good_sets) < 5:
                    # Avoid duplicates
                    if new_set not in good_sets:
                        good_sets.append(new_set)
                else:
                    # Replace a random one? Or just ignore.
                    # Let's replace the worst one in good_sets if new one is better than the worst
                    # Actually, just keeping top 5 by size is good.
                    # For simplicity, if full, replace a random one if new is better than that random one
                    idx = random.randint(0, len(good_sets)-1)
                    if new_size > len(good_sets[idx]):
                        good_sets[idx] = new_set

    return sorted(best_set)
