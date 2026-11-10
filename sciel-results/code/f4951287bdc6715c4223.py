import math
import time
import bisect

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Precompute divisors of N for fast lookup
    divisors = set()
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    divisors_list = sorted(divisors)
    
    best_factors = [N]
    best_max_val = N
    
    def dfs(target_k, current_prod, last_factor, depth, path):
        nonlocal best_max_val, best_factors
        
        # Time check: keep a safety margin
        if time.time() - start_time > 80:
            return
        
        if depth == target_k:
            if current_prod == N:
                # The factors are in path
                max_f = max(path)
                if max_f < best_max_val:
                    best_max_val = max_f
                    best_factors = list(path)
            return
        
        remaining = target_k - depth
        
        if N % current_prod != 0:
            return
        target_rem = N // current_prod
        
        # Minimum possible product for the remaining 'remaining' factors
        # They must be distinct and > last_factor
        min_prod = 1
        for i in range(1, remaining + 1):
            min_prod *= (last_factor + i)
            
        if target_rem < min_prod:
            return
        
        # If remaining == 1, the next factor must be target_rem
        if remaining == 1:
            x = target_rem
            if x > last_factor and x > 12:
                path.append(x)
                dfs(target_k, current_prod * x, x, depth + 1, path)
                path.pop()
            return
        
        lower_bound = max(last_factor + 1, 13)
        
        # Estimate upper bound for x
        # x * (x+1) * ... * (x+remaining-1) <= target_rem
        # x^remaining <= target_rem => x <= target_rem^(1/remaining)
        # Add a margin
        x_upper = int(target_rem ** (1.0 / remaining)) + 10
        
        # Find divisors of N in [lower_bound, x_upper] that also divide target_rem
        # Use bisect to find the range in divisors_list
        left_idx = bisect.bisect_left(divisors_list, lower_bound)
        right_idx = bisect.bisect_right(divisors_list, x_upper)
        
        for idx in range(left_idx, right_idx):
            if time.time() - start_time > 80:
                return
            
            x = divisors_list[idx]
            
            if x > last_factor and x > 12 and target_rem % x == 0:
                path.append(x)
                dfs(target_k, current_prod * x, x, depth + 1, path)
                path.pop()

    # Try all k from 7 down to 2
    # Minimizing the largest factor usually favors more factors (smaller individual factors).
    # But distinctness constraint makes it harder.
    # Let's try k=7 down to 2.
    
    for k in range(7, 1, -1):
        dfs(k, 1, 12, 0, [])
        
    return best_factors

# EVOLVE-BLOCK-START
# The above code is the logic. I will wrap it in the required format.
# EVOLVE-BLOCK-END
