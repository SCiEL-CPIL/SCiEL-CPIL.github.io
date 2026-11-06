import time
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Precompute harmonic numbers
    max_h = 10000
    H = [0.0] * (max_h + 1)
    for i in range(1, max_h + 1):
        H[i] = H[i-1] + 1.0 / i
        
    def harmonic_sum(lo, hi):
        if lo > hi:
            return 0.0
        if hi > max_h:
            return math.log(hi) - math.log(lo - 1)
        return H[hi] - (H[lo - 1] if lo > 1 else 0.0)

    import math # for log in harmonic_sum fallback

    def can_solve(num, den, min_d, limit):
        """
        Returns True if num/den can be represented as sum of distinct reciprocals
        from [min_d, limit].
        """
        # Time check
        if time.time() - start_time > time_limit:
            return False
        
        if num == 0:
            return True
        
        if num < 0:
            return False
            
        # Lower bound for d
        lower_bound = (den + num - 1) // num
        if lower_bound < min_d:
            lower_bound = min_d
            
        if lower_bound > limit:
            return False
            
        # Max possible sum check
        max_possible = harmonic_sum(lower_bound, limit)
        if max_possible < num / den - 1e-15:
            return False
            
        # Iterate d
        for d in range(lower_bound, limit + 1):
            # Check if 1/d <= num/den
            if num * d < den:
                continue
                
            next_num = num * d - den
            next_den = den * d
            g = gcd(next_num, next_den)
            if g > 1:
                next_num //= g
                next_den //= g
                
            # Prune next step
            if next_num > 0:
                # Max sum from d+1 to limit
                max_rem = harmonic_sum(d + 1, limit)
                if max_rem < next_num / next_den - 1e-15:
                    continue
                    
            if can_solve(next_num, next_den, d + 1, limit):
                return True
                
        return False

    # Binary search for minimal limit
    lo = 100
    hi = 2000
    best_limit = 2000
    
    while lo <= hi:
        if time.time() - start_time > time_limit - 1.0:
            break
        mid = (lo + hi) // 2
        if can_solve(5, 31, 1, mid):
            best_limit = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # We found best_limit. Now we need to construct the actual solution.
    # The task requires returning the list of denominators.
    
    def find_solution(num, den, min_d, limit):
        if time.time() - start_time > time_limit:
            return None
            
        if num == 0:
            return []
            
        if num < 0:
            return None
            
        lower_bound = (den + num - 1) // num
        if lower_bound < min_d:
            lower_bound = min_d
            
        if lower_bound > limit:
            return None
            
        max_possible = harmonic_sum(lower_bound, limit)
        if max_possible < num / den - 1e-15:
            return None
            
        for d in range(lower_bound, limit + 1):
            if num * d < den:
                continue
                
            next_num = num * d - den
            next_den = den * d
            g = gcd(next_num, next_den)
            if g > 1:
                next_num //= g
                next_den //= g
                
            if next_num > 0:
                max_rem = harmonic_sum(d + 1, limit)
                if max_rem < next_num / next_den - 1e-15:
                    continue
                    
            res = find_solution(next_num, next_den, d + 1, limit)
            if res is not None:
                return [d] + res
                
        return None
        
    # Try to find solution with best_limit
    # It's possible can_solve returned True but find_solution fails due to time or bugs
    # So we try a few limits around best_limit
    for limit in range(best_limit, best_limit + 10):
        sol = find_solution(5, 31, 1, limit)
        if sol is not None:
            return sol
            
    # Fallback to known solution
    return [8, 28, 1736]
