import time
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # LCM of all divisors: 2*3*5*7*11*13 = 30030
    LCM = 30030
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sort divisors descending for better pruning in DFS
    indexed_divs = sorted(enumerate(divisors), key=lambda x: x[1], reverse=True)
    
    best_span = float('inf')
    best_reps = [0] * n
    
    def dfs(idx, chosen_set, current_max, L):
        nonlocal best_span, best_reps
        
        if time.time() > deadline:
            return False
            
        if idx == n:
            span = current_max - L
            if span < best_span:
                best_span = span
                # Reconstruct in original order
                temp = [0] * n
                for i, (orig_idx, d) in enumerate(indexed_divs):
                    # Need to find which value was assigned to this divisor
                    # We track via chosen values list
                    pass
                # Actually, let's track assignments properly
                best_reps = assignments[:]
            return True
        
        orig_idx, d = indexed_divs[idx]
        
        # Start from smallest multiple >= L
        k = (L + d - 1) // d
        val = k * d
        
        # Try each multiple in the window
        while val - L < best_span:  # prune: can't exceed current best span
            if val not in chosen_set:
                chosen_set.add(val)
                new_max = max(current_max, val)
                assignments[orig_idx] = val
                if dfs(idx + 1, chosen_set, new_max, L):
                    return True
                chosen_set.discard(val)
                assignments[orig_idx] = 0
            val += d
        
        return False
    
    assignments = [0] * n
    
    # Search over L in one LCM period
    # But we can be smarter: L should be a multiple of at least one divisor
    # for the minimum to be tight
    # Generate candidate L values that are multiples of divisors in [1, LCM]
    candidate_L = set()
    for d in divisors:
        for m in range(d, LCM + 1, d):
            candidate_L.add(m)
    
    candidate_L = sorted(candidate_L)
    
    for L in candidate_L:
        if time.time() > deadline:
            break
        
        if best_span <= 9:
            break
            
        chosen_set = set()
        assignments = [0] * n
        dfs(0, chosen_set, L, L)
        
        # Also try L-1, L+1, etc. to catch cases where min isn't exactly a multiple
        # Actually, the min of the solution IS a multiple of some divisor (the one assigned to it)
        # So we don't need to offset L
    
    # If DFS didn't find anything good, fallback to greedy scan
    if best_span > 50 or best_reps[0] == 0:
        def next_multiple(M, d):
            return ((M + d - 1) // d) * d
        
        for L in candidate_L:
            if time.time() > deadline:
                break
            if best_span <= 9:
                break
            
            candidates = []
            valid = True
            for d in divisors:
                x = next_multiple(L, d)
                if x > max_val:
                    valid = False
                    break
                candidates.append(x)
            
            if not valid:
                continue
            
            if len(set(candidates)) != n:
                continue
            
            span = max(candidates) - min(candidates)
            if span < best_span:
                best_span = span
                best_reps = candidates[:]
    
    # Final fallback
    if best_reps[0] == 0:
        best_reps = divisors[:]
    
    return [int(x) for x in best_reps]
