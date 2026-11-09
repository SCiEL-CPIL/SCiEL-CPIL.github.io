import itertools

def construct():
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    # Split divisors into two halves
    mid = len(divisors) // 2
    d1 = divisors[:mid]
    d2 = divisors[mid:]
    
    # Helper to generate all combinations of size k from a list
    def gen_combos(lst, k):
        if k == 0:
            yield ()
        else:
            for c in itertools.combinations(lst, k):
                yield c

    for k in range(1, 6):
        found = False
        # Try all splits i from 0 to k
        for i in range(0, k + 1):
            j = k - i
            # Generate sums for d1 of size i
            # Store sum -> combo
            sums_d1 = {}
            for c1 in gen_combos(d1, i):
                s1 = sum(c1)
                # We only need one combo per sum? 
                # If there are multiple, any is fine.
                if s1 not in sums_d1:
                    sums_d1[s1] = c1
            
            # Iterate combos for d2 of size j
            for c2 in gen_combos(d2, j):
                s2 = sum(c2)
                need = target - s2
                if need in sums_d1:
                    c1 = sums_d1[need]
                    # Verify distinctness? 
                    # d1 and d2 are disjoint, so elements are distinct.
                    # Verify sum? Yes, by construction.
                    return list(c1) + list(c2)
                    
        # If not found for this k, continue to next k
        
    # Fallback
    return [2520, 1680, 720, 56, 3]
