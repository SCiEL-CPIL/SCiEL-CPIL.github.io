import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for 60s limit
    
    # Theoretical minimum k for denominators >= 5 is 7.
    # We search for k = 7, 8, 9, 10.
    # If found, return immediately.
    
    def dfs(rem, min_d, k_left, path):
        if time.time() > deadline:
            return None
        
        if k_left == 0:
            if rem == 0:
                return path[:]
            return None
        
        if k_left == 1:
            # 1/d = rem => d = 1/rem
            if rem.numerator == 1:
                d = rem.denominator
                if d >= min_d and d <= 10000000:
                    return path + [d]
            return None
        
        # Pruning:
        # 1. 1/d <= rem  => d >= 1/rem
        # 2. rem - 1/d must be formable by k_left-1 terms starting from d+1.
        #    Max sum of k_left-1 terms starting from d+1 is sum(1/(d+1+i) for i in range(k_left-1))
        #    If rem - 1/d > max_sum, then d is too large? 
        #    No, if d increases, 1/d decreases, so rem - 1/d increases.
        #    And max_sum decreases.
        #    So if rem - 1/d > max_sum, then for any larger d, rem - 1/d will be even larger 
        #    and max_sum even smaller. So we can break.
        
        # Lower bound for d:
        # d must be at least min_d.
        # Also 1/d <= rem => d >= ceil(1/rem).
        d_min = max(min_d, int(1 / rem) + 1)
        
        # If d_min is such that 1/d_min > rem, we need larger d.
        # The loop handles this via the break condition or start point.
        
        # Upper bound for d:
        # We need rem - 1/d <= sum_{j=0}^{k_left-2} 1/(d+1+j)
        # Let S(d) = sum_{j=0}^{k_left-2} 1/(d+1+j)
        # We need rem - 1/d <= S(d)
        # rem <= 1/d + S(d)
        # Let f(d) = 1/d + S(d). This is decreasing in d.
        # So we need f(d) >= rem.
        # If f(d) < rem, then d is too small? 
        # Wait. If d is small, 1/d is large, S(d) is large. f(d) is large.
        # If d is large, f(d) is small.
        # So if f(d) < rem, we need smaller d? But we are iterating d upwards from d_min.
        # So if f(d_min) < rem, then no solution exists for this branch?
        # Actually, if f(d) < rem for the *smallest* possible d, then the max possible sum 
        # of k_left terms starting from min_d is less than rem. So we can return None immediately.
        
        # Let's check if max possible sum from min_d is enough.
        max_possible_sum = sum(1.0 / (min_d + i) for i in range(k_left))
        if max_possible_sum < rem - 1e-12:
            return None
            
        # Now iterate d
        # We need an upper bound for d to avoid infinite loop if no solution.
        # From rem <= 1/d + S(d), since S(d) is small for large d, roughly rem <= 1/d + (k_left-1)/(d+1).
        # This implies d is not arbitrarily large.
        # A safe upper bound: d_max = int(rem * k_left) + 10? 
        # Actually, if we pick d too large, rem - 1/d ~ rem.
        # We need to form rem with k_left-1 terms starting from d+1.
        # The smallest term is 1/(d+1). The sum is approx (k_left-1)/(d+1).
        # So we need rem <= (k_left-1)/(d+1) => d+1 <= (k_left-1)/rem => d <= (k_left-1)/rem - 1.
        # This is a good upper bound.
        
        d_max = int((k_left - 1) / rem) + 1
        if d_max < d_min:
            return None
            
        for d in range(d_min, d_max + 1):
            # Check if 1/d <= rem
            if Fraction(1, d) > rem:
                continue
                
            new_rem = rem - Fraction(1, d)
            
            # Pruning: Can we form new_rem with k_left-1 terms starting from d+1?
            # Max sum of k_left-1 terms starting from d+1:
            max_next_sum = sum(1.0 / (d + 1 + i) for i in range(k_left - 1))
            if new_rem > max_next_sum + 1e-12:
                # If new_rem is too large, we need smaller denominators.
                # But we are increasing d. As d increases, new_rem increases (1/d decreases).
                # And max_next_sum decreases.
                # So if it's too large now, it will be too large for all larger d.
                break
                
            path.append(d)
            res = dfs(new_rem, d + 1, k_left - 1, path)
            if res:
                return res
            path.pop()
            
        return None

    # Try small k first
    for k in range(7, 13):
        if time.time() > deadline - 5.0:
            break
        sol = dfs(Fraction(1, 1), 5, k, [])
        if sol:
            return sol

    # Fallback to DP if DFS didn't find a small solution quickly
    # The DP finds a solution with k=20 reliably.
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
        if time.time() > deadline:
            break
            
    if target in reached:
        return reached[target]
    raise RuntimeError("no baseline representation")
