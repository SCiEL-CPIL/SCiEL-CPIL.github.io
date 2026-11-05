import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Base set: [171, 256]
    base_S = list(range(171, n + 1))
    best_S = set(base_S)
    best_size = len(base_S)
    
    candidates = list(range(1, 171))
    
    # Precompute sums for base set
    def compute_sums(S_list):
        sums = set()
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            for j in range(i, L):
                sums.add(b + S_list[j])
        return sums

    base_sums = compute_sums(base_S)
    
    # Function to evaluate a specific order of adding candidates
    def evaluate_order(order):
        current_S = base_S[:]
        current_sums = set(base_sums)
        
        for x in order:
            # Check if x divides any sum in current_sums
            # x must be < b <= c, so x is 'a'.
            # We check if there exists k such that k*x is in current_sums.
            # Min sum in current_sums is at least 2*min(current_S).
            # Max sum is 512.
            
            # Optimization: Check multiples of x
            # Start from 2*x? No, min sum might be larger.
            # Just iterate multiples up to 512.
            conflict = False
            # The smallest possible sum in current_sums involving x as 'a'
            # is x + min(current_S). But current_sums contains sums of pairs in current_S.
            # When we check x, we are asking: does x divide (b+c) for any b,c in current_S?
            # So we just check if any multiple of x is in current_sums.
            # The multiples must be >= min(current_sums) and <= max(current_sums).
            # But checking all multiples up to 512 is fast enough (512/x iterations).
            
            # Faster: 
            # min_mult = (min(current_sums) + x - 1) // x
            # for m in range(min_mult * x, 513, x):
            #    if m in current_sums: conflict = True; break
            
            # Even faster: just iterate
            for m in range(2 * x, 513, x):
                if m in current_sums:
                    conflict = True
                    break
            
            if not conflict:
                # Add x
                # New sums to add: x + y for all y in current_S, and 2*x
                # Note: current_S is the set BEFORE adding x.
                for y in current_S:
                    current_sums.add(x + y)
                current_sums.add(2 * x)
                current_S.append(x)
        
        return current_S

    # Heuristics for ordering
    # 1. Random
    # 2. Descending (larger small-numbers first)
    # 3. Ascending
    # 4. By number of divisors (fewer divisors first)
    
    def num_divisors(x):
        count = 0
        i = 1
        while i * i <= x:
            if x % i == 0:
                count += 1
                if i * i != x:
                    count += 1
            i += 1
        return count

    candidates_by_divisors = sorted(candidates, key=num_divisors)
    candidates_desc = sorted(candidates, reverse=True)
    candidates_asc = sorted(candidates)
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Cycle through heuristics
        if restarts % 4 == 1:
            order = candidates[:]
            random.shuffle(order)
        elif restarts % 4 == 2:
            order = candidates_desc[:]
        elif restarts % 4 == 3:
            order = candidates_asc[:]
        else:
            order = candidates_by_divisors[:]
        
        result_S = evaluate_order(order)
        if len(result_S) > best_size:
            best_size = len(result_S)
            best_S = set(result_S)
            
        # Early exit if we found a very large set? 
        # Max possible is 256. 
        # If we get > 150, it's a huge jump.
        # But let's run to deadline to be safe.
        
    return list(best_S)
