import time

def _check_divisibility(sums_set):
    """Check that no two distinct elements in sums_set divide one another.
    Returns True if valid, False otherwise."""
    if len(sums_set) <= 1:
        return True
    sums_list = sorted(sums_set)
    max_sum = sums_list[-1]
    for s in sums_list:
        # Check if any multiple of s (2s, 3s, ...) is in the set
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_set = []
    best_sums = set()
    
    # Candidate numbers from 256 down to 1
    candidates = range(256, 0, -1)
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        # Calculate new sums if x is added
        # new_sums = {s + x for s in best_sums} | {x}
        new_sums = set()
        for s in best_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        
        # Check 1: Distinctness
        # No element in new_sums can be in best_sums
        if new_sums & best_sums:
            continue
        
        # Check 2: Divisibility on the combined set
        # We need to check the full set best_sums | new_sums
        # But we can be smarter: only new pairs matter
        # Pairs to check:
        #   - new_sums vs new_sums (internal)
        #   - new_sums vs best_sums (cross)
        # Pairs within best_sums are already valid.
        
        # Efficient check: for the combined set, use multiple-based approach
        # But we can optimize: only check divisibility involving at least one new sum
        
        valid = True
        
        # Check: for each n in new_sums, no other sum in (best_sums | new_sums) is a multiple of n
        # And for each s in best_sums, no sum in new_sums is a multiple of s
        # Combined: for each element in the combined set, check multiples, but only 
        # need to verify pairs where at least one is new.
        
        # Simplest efficient approach: build combined set and run _check_divisibility
        # But that's O(N * max/min) which might be similar to before.
        # Let's try a targeted check.
        
        all_sums = best_sums | new_sums
        max_sum = max(all_sums)
        
        # For each n in new_sums, check if any multiple of n is in all_sums
        for n in new_sums:
            multiple = 2 * n
            while multiple <= max_sum:
                if multiple in all_sums:
                    valid = False
                    break
                multiple += n
            if not valid:
                break
        
        if not valid:
            continue
        
        # For each s in best_sums, check if any element in new_sums is a multiple of s
        for s in best_sums:
            # Check if any element in new_sums is a multiple of s
            # i.e., is there n in new_sums such that n % s == 0?
            # Equivalent: is there k >= 1 such that k*s is in new_sums?
            # Since s < n for all n in new_sums that are multiples (n > s),
            # k >= 2 if n > s, but k=1 means n=s which is excluded by distinctness.
            # So check k >= 2: 2s, 3s, ... in new_sums?
            # But also k=1: s itself, but s is not in new_sums (distinctness check passed).
            # So just check multiples starting from 2s.
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in new_sums:
                    valid = False
                    break
                multiple += s
            if not valid:
                break
        
        if not valid:
            continue
        
        # If we get here, the set is valid
        # Accept x
        best_set.append(x)
        best_sums = best_sums.union(new_sums)
    
    # Return as list of ints
    return list(best_set)
