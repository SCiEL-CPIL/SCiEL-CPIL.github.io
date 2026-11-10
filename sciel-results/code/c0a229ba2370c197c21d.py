import math
import time

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
        
        # Iterate through divisors of target_rem that are >= lower_bound
        # To optimize, we can find the index in divisors_list
        # But divisors_list is sorted. We need divisors of target_rem, not N.
        # However, if x divides target_rem and target_rem divides N, then x divides N.
        # So we can filter divisors of N.
        
        # Estimate upper bound for x
        # x * (x+1) * ... * (x+remaining-1) <= target_rem
        # x^remaining <= target_rem => x <= target_rem^(1/remaining)
        # Add a margin
        x_upper = int(target_rem ** (1.0 / remaining)) + 10
        
        # Filter divisors of N that are in [lower_bound, x_upper]
        # Since divisors_list is sorted, we can use binary search or just filter
        # Given N is small enough, iterating all divisors might be fast if we precompute them?
        # No, we need divisors of target_rem specifically.
        # But checking `target_rem % x == 0` is fast.
        
        # Let's iterate from lower_bound to x_upper and check divisibility.
        # This is what the original code did.
        # Can we do better?
        # We can iterate through divisors of N that are >= lower_bound and <= x_upper.
        
        # Let's find the relevant divisors from the precomputed list
        # divisors_list is sorted.
        # We need d in divisors_list such that lower_bound <= d <= x_upper and target_rem % d == 0.
        
        # Since divisors_list has ~150 elements, we can just loop through it.
        # But we need to ensure we don't miss any.
        # Actually, iterating range(lower_bound, x_upper+1) is fine if the range is small.
        # For small remaining, x_upper is small.
        # For large remaining, x_upper is small.
        # The worst case is remaining=2, x_upper ~ sqrt(N) ~ 22000.
        # 22000 iterations is fine.
        
        for x in range(lower_bound, x_upper + 1):
            if time.time() - start_time > 80:
                return
            
            if target_rem % x != 0:
                continue
            
            path.append(x)
            dfs(target_k, current_prod * x, x, depth + 1, path)
            path.pop()

    # Try all k from 1 to 7
    # k=1 is trivial [N]
    # We start from k=7 down to 2 to find better solutions faster?
    # Or k=2 to 7?
    # Minimizing the largest factor usually favors more factors (smaller individual factors).
    # But distinctness constraint makes it harder.
    # Let's try k=7 down to 2.
    
    for k in range(7, 1, -1):
        dfs(k, 1, 12, 0, [])
        
    return best_factors

# EVOLVE-BLOCK-START
# The above code is the logic. I will wrap it in the required format.
# EVOLVE-BLOCK-END
