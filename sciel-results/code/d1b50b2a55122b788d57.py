import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    deadline = 80.0
    
    target = factorial(12)  # 479001600
    
    def get_divisors(n):
        divs = set()
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)
    
    all_divs = get_divisors(target)
    
    # Prime factorization of target
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    def can_decompose(M, time_limit):
        pool = [d for d in all_divs if 12 < d <= M]
        if not pool:
            return None
        if target in pool:
            return [target]
        
        n_pool = len(pool)
        
        def dfs(idx, remaining, count, chosen):
            if time.time() > time_limit:
                return None
            
            if remaining == 1:
                return chosen[:]
            
            if count >= 12:
                return None
            
            if idx >= n_pool:
                return None
            
            # Pruning: remaining must be >= pool[idx]
            if remaining < pool[idx]:
                return None
            
            # Pruning: if remaining is prime, it must be in pool[idx:]
            # (but checking primality is expensive; skip for now)
            
            # Pruning: max possible product of (12-count) factors from pool[idx:]
            # is at most M^(12-count), but more precisely the product of the largest
            # (12-count) elements. If that's < remaining, fail.
            slots_left = 12 - count
            if slots_left <= n_pool - idx:
                # Product of largest slots_left elements from pool[idx:]
                # This is expensive to compute each time; skip
                pass
            
            for i in range(idx, n_pool):
                d = pool[i]
                if d > remaining:
                    break
                if remaining % d != 0:
                    continue
                
                new_remaining = remaining // d
                if new_remaining == 1:
                    return chosen + [d]
                
                # Pruning: new_remaining must be >= pool[i+1] if there are more elements
                if i + 1 < n_pool and new_remaining < pool[i + 1]:
                    continue
                
                # Pruning: new_remaining must be >= 1 (always true)
                # Pruning: if new_remaining > 1, we need at least one more factor
                if count + 1 >= 12:
                    continue
                
                result = dfs(i + 1, new_remaining, count + 1, chosen + [d])
                if result is not None:
                    return result
            return None
        
        return dfs(0, target, 0, [])
    
    # Binary search for minimum M
    lo = 13
    hi = target
    best_result = [target]
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        result = can_decompose(mid, deadline - 5)
        if result is not None:
            best_result = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_result
