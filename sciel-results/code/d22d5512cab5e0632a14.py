import time
from math import factorial, isqrt
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin under 90s kill
    
    N = factorial(12)  # 479001600
    
    # Generate all divisors of N that are > 12
    divs = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            if i > 12:
                divs.append(i)
            other = N // i
            if other != i and other > 12:
                divs.append(other)
        i += 1
    
    # Ensure N itself is in the list (it is, as 1*N, but 1 is not >12, so we add N explicitly if not present)
    # Actually, if N > 12, it will be in divs. 479001600 > 12.
    
    divs.sort()
    divs_set = set(divs)
    
    best_max = N + 1
    best_factors = []
    
    # Max number of factors allowed is 12
    MAX_FACTORS = 12
    
    def dfs(remaining, last_factor, current_factors, current_max, depth):
        nonlocal best_max, best_factors
        
        # If we found a complete partition (remaining is 1)
        if remaining == 1:
            if current_max < best_max:
                best_max = current_max
                best_factors = list(current_factors)
            return
        
        # Pruning 1: If we have used too many factors
        if depth >= MAX_FACTORS:
            return
            
        # Pruning 2: Check if it's even possible to complete the product with distinct integers > last_factor
        # The minimum product we can achieve with k distinct integers starting from last_factor+1
        # We don't know k, but we can check if the product of the next few integers exceeds remaining.
        # A simple check: if (last_factor + 1) * (last_factor + 2) * ... > remaining, stop.
        # Since we are picking divisors, we can just iterate.
        
        # Pruning 3: The next factor must be at least last_factor + 1.
        # If last_factor + 1 >= best_max, we can stop because any new factor will be >= best_max.
        # Actually, if we pick a factor d, the new max is max(current_max, d).
        # If d >= best_max, it won't improve the best_max.
        # So we only care about factors d < best_max.
        
        # Find the first divisor > last_factor
        start_idx = bisect.bisect_right(divs, last_factor)
        
        # We also don't need to consider divisors >= best_max because they won't improve the solution
        # Wait, if we are in a branch where current_max is already close to best_max, 
        # picking a factor d where max(current_max, d) >= best_max is useless.
        # So we can cap the search for d to d < best_max.
        
        for i in range(start_idx, len(divs)):
            d = divs[i]
            
            # If d is too large to improve best_max
            if d >= best_max:
                break
                
            # If d doesn't divide remaining
            if remaining % d != 0:
                continue
                
            # Calculate new max
            new_max = current_max if current_max > d else d
            
            # If new_max is already >= best_max, no point continuing this branch
            # (Since d is increasing, subsequent d's will be larger or equal, so new_max won't decrease)
            if new_max >= best_max:
                 # Actually, if current_max >= best_max, we would have pruned earlier.
                 # If d >= best_max, we break.
                 # If current_max < best_max but new_max >= best_max, it means d >= best_max (handled above) 
                 # or current_max >= best_max (handled by invariant).
                 pass
            
            # Recurse
            current_factors.append(d)
            dfs(remaining // d, d, current_factors, new_max, depth + 1)
            current_factors.pop()
            
            # Time check
            if time.time() - t0 > DEADLINE:
                return

    # Start DFS
    dfs(N, 12, [], 12, 0)
    
    # If no solution found better than N, return [N]
    if best_factors:
        return sorted(best_factors)
    else:
        return [N]
