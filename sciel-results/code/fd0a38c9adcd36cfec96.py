import time

def construct():
    deadline = time.time() + 80
    TARGET = 123457
    
    # Precompute largest prime factor for all numbers up to TARGET using a modified sieve.
    # lp[n] will store the largest prime factor of n.
    lp = [0] * (TARGET + 1)
    lp[1] = 1
    for i in range(2, TARGET + 1):
        if lp[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lp[j] = i  # i is a prime factor of j; since we iterate i in increasing order, the last assignment is the largest prime factor
    
    # We want to find a, b > 0 with a + b = TARGET such that max(lp[a], lp[b]) is minimized.
    # Let M = max(lp[a], lp[b]). We want to minimize M.
    # This is equivalent to finding the smallest prime p (or 1) such that there exists a
    # decomposition where both a and b have all prime factors <= p.
    
    # Strategy:
    # 1. Identify the set of candidate "bounds" B. These are the unique values in lp[1..TARGET-1].
    # 2. Sort these unique values in ascending order.
    # 3. For each candidate bound B in sorted order:
    #    a. Check if there exists an a in [1, TARGET-1] such that lp[a] <= B and lp[TARGET-a] <= B.
    #    b. If such an a exists, then B is the minimal possible max prime factor (since we check in increasing order).
    #    c. Return [a, TARGET-a].
    
    # Optimization:
    # Instead of iterating over all a for each B (which is O(N * U) where U is unique LPs),
    # we can be smarter.
    # Let S_B = {x in [1, TARGET-1] | lp[x] <= B}.
    # We need to check if S_B intersects with (TARGET - S_B).
    # i.e., does there exist x in S_B such that TARGET - x is also in S_B?
    
    # Since N is small (123457), we can just do the check efficiently.
    # For a given B, we can create a boolean array or set of valid numbers.
    # But creating a set for each B is expensive if we do it from scratch.
    
    # Alternative approach:
    # Sort the numbers 1..TARGET-1 by their lp value.
    # Let's group numbers by their lp value.
    # lp_groups[v] = list of numbers x such that lp[x] == v.
    
    # Then, for a bound B, the set of valid numbers is the union of lp_groups[v] for all v <= B.
    # We can maintain a set `valid_set` that grows as B increases.
    
    # Steps:
    # 1. Get unique lp values, sorted.
    # 2. Initialize valid_set = empty.
    # 3. Iterate through unique lp values B in sorted order:
    #    a. Add all numbers x with lp[x] == B to valid_set.
    #    b. Check if there exists a in valid_set such that TARGET - a is in valid_set.
    #       To do this efficiently, we can just iterate over the newly added numbers?
    #       No, because the partner might have been added earlier.
    #       So we need to check if the *entire* valid_set has a pair.
    #       However, if a pair existed in the previous step (with smaller B), we would have already found it and returned.
    #       So we only need to check if the *new* additions create a pair with *any* element in the current valid_set.
    #       Actually, simpler: just check if there exists a in valid_set such that TARGET-a in valid_set.
    #       Since valid_set grows, we can just check the condition.
    #       But iterating the whole set each time is O(N^2) in the worst case.
    
    # Let's refine the check:
    # When we add a new batch of numbers (those with lp == B), let's call this batch `new_numbers`.
    # For each a in `new_numbers`:
    #    b = TARGET - a
    #    If b > 0 and b in valid_set:
    #        We found a pair. Return it.
    # Note: We must ensure b is in valid_set. Since we are adding `new_numbers` to `valid_set` *before* checking?
    # If we add first, then b could be in `new_numbers` as well. That's fine, as long as b is in valid_set.
    # So:
    # 1. Add `new_numbers` to `valid_set`.
    # 2. For each a in `new_numbers`:
    #    b = TARGET - a
    #    if b in valid_set:
    #       return [a, b]
    
    # This works because:
    # - If a pair (a, b) exists with max(lp[a], lp[b]) <= B, then both a and b are in valid_set.
    # - When we reached the bound B_max = max(lp[a], lp[b]), both a and b were added to valid_set.
    # - Suppose lp[a] <= lp[b]. Then b is added at step lp[b]. a was added at step lp[a] <= lp[b].
    # - At step lp[b], we add b. We check if TARGET - b = a is in valid_set. It is, because a was added earlier.
    # - So we will find it.
    # - What if lp[a] == lp[b]? Then both are added in the same batch.
    #   We add them to valid_set first. Then we iterate over the batch.
    #   If we pick a, we check if b is in valid_set. It is (just added).
    #   So we find it.
    
    # Complexity:
    # - Sieve: O(N log log N)
    # - Grouping: O(N)
    # - Main loop: Each number is added to valid_set once. For each addition, we do a set lookup O(1).
    #   Total O(N).
    
    # Implementation:
    
    # 1. Compute lp array (already done above)
    
    # 2. Find unique lp values for numbers 1 to TARGET-1
    unique_lps = sorted(set(lp[1:TARGET]))
    
    # 3. Group numbers by lp value
    # We only care about numbers 1 to TARGET-1
    groups = {}
    for x in range(1, TARGET):
        val = lp[x]
        if val not in groups:
            groups[val] = []
        groups[val].append(x)
    
    valid_set = set()
    
    for B in unique_lps:
        if time.time() > deadline - 2:
            break
            
        # Get numbers with lp == B
        new_numbers = groups.get(B, [])
        
        # Add them to valid_set
        for x in new_numbers:
            valid_set.add(x)
        
        # Check if any new number forms a pair with a number in valid_set
        for a in new_numbers:
            b = TARGET - a
            if b in valid_set:
                # Found a valid pair
                # Ensure a, b are positive (they are, by construction of range 1..TARGET-1)
                # The max prime factor is at most B
                return [a, b]
    
    # Fallback (should not happen)
    return [1, TARGET - 1]
