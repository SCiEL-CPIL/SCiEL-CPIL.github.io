import time
from itertools import combinations

def construct():
    # Deadline guard: 90s limit, use 85s for safety
    start_time = time.time()
    deadline = start_time + 85
    
    p = 101
    denominators = list(range(1, 11))
    # Precompute inverses
    invs = [pow(x, -1, p) for x in denominators]
    
    # We want to find the smallest k such that all residues 0..100 are covered
    # by sums of inverses of k distinct denominators.
    
    # Store the best assignment found so far (initially None)
    best_assignment = None
    best_max_len = 11 # Max possible is 10, so 11 is "inf"
    
    # Iterate k from 1 to 10
    for k in range(1, 11):
        # Check time
        if time.time() > deadline:
            break
            
        # Map residue -> one subset (as a tuple of denominators)
        coverage = {}
        
        # Iterate all combinations of size k
        for combo in combinations(denominators, k):
            # Calculate sum of inverses
            s = 0
            for x in combo:
                s += invs[x-1]
            s %= p
            
            # If this residue hasn't been covered yet, store this combo
            if s not in coverage:
                coverage[s] = list(combo)
                
        # Check if all residues are covered
        if len(coverage) == p:
            # We found a valid assignment with max length k
            # Since we iterate k from 1 up, this is the minimal k
            best_assignment = coverage
            best_max_len = k
            break
            
    # If no k worked (should not happen as k=10 covers all), 
    # fallback to a brute force first-found approach similar to seed but optimized
    if best_assignment is None:
        coverage = {}
        for k in range(1, 11):
            for combo in combinations(denominators, k):
                s = sum(invs[x-1] for x in combo) % p
                if s not in coverage:
                    coverage[s] = list(combo)
        best_assignment = coverage

    # Construct the final list of 101 lists
    result = []
    for r in range(p):
        if r in best_assignment:
            result.append(best_assignment[r])
        else:
            # Should not happen if coverage was complete, but safe fallback
            # Find any subset that sums to r
            found = False
            for k in range(1, 11):
                for combo in combinations(denominators, k):
                    s = sum(invs[x-1] for x in combo) % p
                    if s == r:
                        result.append(list(combo))
                        found = True
                        break
                if found:
                    break
            if not found:
                # Absolute fallback, shouldn't reach here
                result.append([1])
                
    return result
