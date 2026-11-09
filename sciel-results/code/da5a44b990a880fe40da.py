import itertools
import time

def construct():
    # LCM of 1..22
    # 1..22 LCM is 232792560
    L = 232792560
    w = [0] * 23
    for i in range(1, 23):
        w[i] = L // i
        
    def is_dissociated(indices):
        sums = {0}
        for i in indices:
            wi = w[i]
            new_sums = set()
            for s in sums:
                new_sums.add(s + wi)
            # Check if new_sums intersects sums
            # If yes, collision
            # Actually, we just need to check if the union size is correct
            # But computing union is O(N)
            # Better: check intersection
            if sums & new_sums:
                return False
            sums |= new_sums
        return True

    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Try size 12
    # To make it faster, we can check specific promising sets first
    # But let's just iterate
    # Optimization: Pre-filter? No.
    
    # We'll iterate combinations of size 12
    # If we find one, return it
    found_12 = None
    for combo in itertools.combinations(range(1, 23), 12):
        if time.time() > deadline:
            break
        if is_dissociated(combo):
            found_12 = list(combo)
            break
            
    if found_12:
        return found_12
        
    # Try size 11
    found_11 = None
    for combo in itertools.combinations(range(1, 23), 11):
        if time.time() > deadline:
            break
        if is_dissociated(combo):
            found_11 = list(combo)
            break
            
    if found_11:
        return found_11
        
    # Fallback: Greedy or Seed
    # Seed is [1, 2, 3, 5, 7, 11, 13, 17, 19] (size 9)
    # Let's try to find a size 10 quickly
    found_10 = None
    for combo in itertools.combinations(range(1, 23), 10):
        if time.time() > deadline:
            break
        if is_dissociated(combo):
            found_10 = list(combo)
            break
            
    if found_10:
        return found_10
        
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
