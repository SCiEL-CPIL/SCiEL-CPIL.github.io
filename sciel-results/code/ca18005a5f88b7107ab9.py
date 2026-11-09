import time
from itertools import combinations

def is_sidon_4_fast(subset):
    mask = 0
    # subset is a tuple of 4 distinct integers
    a0, a1, a2, a3 = subset
    sums = (
        a0+a0, a0+a1, a0+a2, a0+a3,
        a1+a1, a1+a2, a1+a3,
        a2+a2, a2+a3,
        a3+a3
    )
    for s in sums:
        bit = 1 << s
        if mask & bit:
            return False
        mask |= bit
    return True

def has_sidon_4_fast(a_set):
    # a_set is a tuple of 12 integers
    for comb in combinations(a_set, 4):
        if is_sidon_4_fast(comb):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Exhaustive search on small ranges
    # We look for a set with NO Sidon 4 subset (max Sidon size 3)
    for R in range(12, 25):
        if time.time() > deadline:
            break
        
        # Optional: skip if combinations count is too high and we have little time?
        # For now, just iterate and check time.
        
        for s in combinations(range(R + 1), 12):
            if time.time() > deadline:
                # Return best effort? 
                # If we haven't found a 3, we can't return a better score.
                # But we must return a valid set.
                # The current 's' is valid, but likely has Sidon 4.
                # Returning baseline is safer if we don't know.
                # However, we can return 's' if we assume it's "randomly" as good as baseline?
                # No, baseline is known 4.
                # Let's just break and go to random or return baseline.
                pass
            
            if not has_sidon_4_fast(s):
                return list(s)
        
        # If we finished this R without finding, continue to next R

    # Fallback: Random search
    import random
    # Try random sets in various ranges
    # We can bias towards smaller ranges
    for _ in range(200000):
        if time.time() > deadline:
            break
        
        # Mix of small and medium ranges
        if random.random() < 0.7:
            r = random.randint(12, 25)
        else:
            r = random.randint(25, 40)
            
        s = tuple(random.sample(range(r + 1), 12))
        
        if not has_sidon_4_fast(s):
            return list(s)
            
    return list(range(12))
