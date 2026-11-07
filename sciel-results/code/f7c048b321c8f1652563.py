import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Pre-compute LCM of all divisors for periodicity
    # LCM(6, 10, 14, 15, 21, 22, 26, 33, 35, 39)
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    LCM = 30030
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_span = float('inf')
    best_reps = [0] * 10
    
    # Sort divisors descending to constrain the search space earlier
    indexed_divs = sorted(enumerate(divisors), key=lambda x: x[1], reverse=True)
    
    # We iterate over L in one period [1, LCM].
    # The pattern of multiples repeats every LCM, so the optimal span
    # will be found within this range (or a shifted version of it, but
    # since we are minimizing span, the relative positions matter, and
    # the "shape" of the solution repeats).
    # Actually, strictly speaking, if the optimal solution is [a, b],
    # then [a+LCM, b+LCM] is also a valid solution with the same span.
    # So we only need to search L in a range that covers all possible
    # "phases" of the multiples. Since min(x_i) must be a multiple of
    # some d_i, and the set of such multiples is periodic with LCM,
    # searching L in [1, LCM] is sufficient to find the optimal span.
    
    for L in range(1, LCM + 1):
        if time.time() > deadline:
            break
            
        # Backtracking search
        def dfs(idx, chosen_vals, current_max):
            nonlocal best_span, best_reps
            
            if time.time() > deadline:
                return
                
            if idx == n:
                span = current_max - L
                if span < best_span:
                    best_span = span
                    temp_reps = [0] * n
                    for i, (orig_idx, val) in enumerate(zip([d[0] for d in indexed_divs], chosen_vals)):
                        temp_reps[orig_idx] = val
                    best_reps = temp_reps[:]
                return
            
            orig_idx, d = indexed_divs[idx]
            
            # Start from the smallest multiple >= L
            k = (L + d - 1) // d
            val = k * d
            
            # Try multiples
            while True:
                # Pruning: if this value makes the span exceed best_span, stop
                if val - L >= best_span:
                    break
                
                if val not in chosen_vals:
                    chosen_vals.append(val)
                    new_max = max(current_max, val)
                    dfs(idx + 1, chosen_vals, new_max)
                    chosen_vals.pop()
                
                val += d
                
        dfs(0, [], L)
        
        # Heuristic: If we found a span of 9, we can't do better
        if best_span <= 9:
            break
            
    # Ensure types are correct
    return [int(x) for x in best_reps]
