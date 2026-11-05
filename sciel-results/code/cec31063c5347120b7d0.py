import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    n = 255
    target = 255
    
    # 255 = 3 * 5 * 17
    # All multiples of 3 in [1, 255] form a valid set of size 85
    S_base = set(x for x in range(1, n + 1) if x % 3 == 0)
    
    best_set = set(S_base)
    best_size = len(best_set)
    
    # Try to improve via greedy local search
    # Build adjacency: two numbers are connected if gcd > 1
    # Precompute for speed
    def gcd(a, b):
        return math.gcd(a, b)
    
    # For a given set S, find candidates outside S that can be added
    # and try to add one at a time (greedy)
    
    def can_add(S, x):
        for a in S:
            if math.gcd(a, x) == 1:
                return False
        return True
    
    # Greedy improvement: repeatedly try to add the best candidate
    # that maximizes the number of new connections (or just any valid add)
    # Since we want max cardinality, we try to add any element that's compatible
    
    improved = True
    while improved:
        improved = False
        # Sort candidates by some heuristic: prefer those with more common factors
        candidates = [x for x in range(1, n + 1) if x not in S_base and can_add(best_set, x)]
        if not candidates:
            break
        # Greedily add candidates one by one
        # Try adding in order of most "compatible" (highest gcd with members)
        def compat_score(x):
            total = 0
            for a in best_set:
                total += math.gcd(a, x)
            return total
        
        candidates.sort(key=compat_score, reverse=True)
        for x in candidates:
            if can_add(best_set, x):
                best_set.add(x)
                improved = True
                break
        if not candidates:
            break
        # Check time
        if time.time() > deadline:
            break
    
    # Verify the result
    if target not in best_set:
        # This shouldn't happen since 255 is in S_base
        best_set.add(target)
    
    # Verify pairwise gcd > 1
    lst = list(best_set)
    valid = True
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if math.gcd(lst[i], lst[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid or len(best_set) < best_size:
        # Fall back to the base set
        return list(S_base)
    
    return list(best_set)
