import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    N = 128
    # 1. Pre-compute all potential violating triples (a, b, c) with a < b < c
    # such that a+b, a+c, b+c are all <= N.
    # If any sum > N, it can't be in the set, so no violation possible.
    candidates = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab = a + b
            if ab > N:
                break
            for c in range(b + 1, N + 1):
                ac = a + c
                bc = b + c
                if ac > N or bc > N:
                    break
                # Check distinctness of a,b,c is guaranteed by loop.
                # The sums might equal a,b,c or each other, which is allowed by problem statement
                # "six displayed values need not be mutually distinct".
                # But a,b,c must be distinct.
                candidates.append((a, b, c, ab, ac, bc))
                
    # 2. Initialize set S and violation counts
    S = set(range(1, N + 1))
    # counts[x] = number of active violations involving x
    counts = [0] * (N + 1)
    # active[i] = True if candidate i is currently a violation
    active = [False] * len(candidates)
    
    # Initial pass to count violations
    for i, (a, b, c, ab, ac, bc) in enumerate(candidates):
        # Since S is initially full, all are in S
        active[i] = True
        for x in (a, b, c, ab, ac, bc):
            counts[x] += 1
            
    # 3. Peeling loop
    while True:
        if time.time() > deadline:
            break
            
        # Find max count
        max_count = -1
        max_val = -1
        for x in S:
            if counts[x] > max_count:
                max_count = counts[x]
                max_val = x
                
        if max_count == 0:
            break # No violations left
            
        # Remove max_val
        S.remove(max_val)
        
        # Update counts for all candidates involving max_val
        # We need to iterate through candidates. To speed this up, 
        # we could pre-compute a list of candidate indices for each value.
        # But N is small, so iterating all candidates is fine.
        for i, (a, b, c, ab, ac, bc) in enumerate(candidates):
            if active[i]:
                # Check if max_val is in this candidate's set
                vals = (a, b, c, ab, ac, bc)
                if max_val in vals:
                    # This violation is now broken
                    active[i] = False
                    for x in vals:
                        if x in S: # Only decrement if still in S? 
                                    # Actually, counts track "involvement in active violations".
                                    # If x is removed, it doesn't matter, but we should keep counts consistent 
                                    # for remaining elements.
                                    # The count for x should be decremented because this violation is no longer active.
                                    counts[x] -= 1
                                    
    # Convert to list of ints
    result = [int(x) for x in S]
    return result
