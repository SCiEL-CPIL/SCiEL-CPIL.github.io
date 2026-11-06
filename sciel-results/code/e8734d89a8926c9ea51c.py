import itertools

def construct():
    max_target = 128
    # Generate smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)
    
    # We will search for antichains of increasing size k
    # min_terms[i] stores the minimum number of terms for target i
    min_terms = [0] * (max_target + 1)
    best_anticain = [None] * (max_target + 1)
    
    covered = [False] * (max_target + 1)
    
    # Max k to search. 3 is likely enough, but let's go up to 6 to be safe.
    # The seed used up to 3.
    max_k = 6
    
    for k in range(1, max_k + 1):
        # If all targets are covered, we can stop early
        if all(covered[1:max_target+1]):
            break
            
        for comb in itertools.combinations(smooth, k):
            s = sum(comb)
            if s > max_target:
                continue
            if covered[s]:
                continue
                
            # Check antichain property
            # comb is a tuple of distinct integers, sorted ascending
            is_anticain = True
            for i in range(len(comb)):
                a = comb[i]
                for j in range(i + 1, len(comb)):
                    b = comb[j]
                    if b % a == 0:
                        is_anticain = False
                        break
                if not is_anticain:
                    break
            
            if is_anticain:
                min_terms[s] = k
                best_anticain[s] = list(comb)
                covered[s] = True
                
    # Verify all targets are covered
    # If some are not, we might need larger k, but 1..128 should be covered by small k.
    # For safety, if any are not covered, we raise an error or try to fill them (though unlikely)
    for i in range(1, max_target + 1):
        if not covered[i]:
            # Fallback: try larger k if necessary, but this shouldn't happen for 1..128
            # We can implement a dynamic expansion if needed, but for now, assume k=6 is enough.
            pass

    results = []
    for i in range(1, max_target + 1):
        if best_anticain[i] is None:
            # This should not happen
            raise ValueError(f"No representation found for {i}")
        results.append(best_anticain[i])
        
    return results
