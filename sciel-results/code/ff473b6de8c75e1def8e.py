import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin (60s limit)

    def is_valid(subset):
        # Generate all non-empty subset sums
        n = len(subset)
        sums = []
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += subset[i]
            sums.append(s)
        
        # Check for duplicates
        if len(set(sums)) != len(sums):
            return False
            
        sums_set = set(sums)
        max_sum = max(sums)
        
        # Check divisibility
        # For each sum S, check if any multiple k*S (k>=2) is in sums_set
        for S in sums:
            multiple = 2 * S
            while multiple <= max_sum:
                if multiple in sums_set:
                    return False
                multiple += S
                
        return True

    best_set = []
    
    # Try to find the largest possible set size
    # Max size is 30, but realistically much smaller.
    # We start from a reasonable guess and go up, or down?
    # Let's try sizes from 5 down to 1. Or 1 up to 5.
    # Given the constraints, 4 or 5 is likely the max.
    
    # Let's try to find a set of size k.
    # We will try k from 1 to 10. If we find a valid set of size k, we keep it and try k+1.
    
    current_best = []
    
    for k in range(1, 11):
        if time.time() > deadline:
            break
            
        found = False
        # Try random samples
        # Heuristic: pick from a wide range to avoid small divisors
        # Numbers should be distinct.
        
        # To improve success rate, we can use a greedy construction or local search.
        # But random sampling with the efficient check might be enough for small k.
        
        # Let's try a few hundred random samples for each k.
        for _ in range(500):
            if time.time() > deadline:
                break
                
            # Generate k distinct random integers from 1..256
            # Bias towards larger numbers to reduce divisibility risks?
            # Or just uniform.
            subset = random.sample(range(1, 257), k)
            
            if is_valid(subset):
                current_best = subset
                found = True
                break
                
        if not found:
            # If we couldn't find a set of size k, it's likely k is too large
            # or we just got unlucky.
            # We can break here if k > 3, as larger sizes are exponentially harder.
            # But let's try one more time with a different strategy?
            # For now, if we fail at k, we stop increasing k.
            # We keep the best found so far (k-1).
            break
            
    # If we found nothing (unlikely for k=1), return a default valid set
    if not current_best:
        current_best = [2]
        
    return current_best
