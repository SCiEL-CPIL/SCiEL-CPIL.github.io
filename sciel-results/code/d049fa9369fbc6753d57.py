import time
import bisect
from math import factorial, isqrt

_START = time.time()
_DEADLINE = _START + 55.0  # Safety margin before 60s limit

def construct():
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = set()
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.add(val)
    
    valid_divs = sorted(divisors)
    n_divs = len(valid_divs)
    divisor_set = set(valid_divs)  # For O(1) lookup
    
    # Try to find a good initial solution quickly
    # Known solution: [22, 21, 25, 18, 36, 64] with max 64
    # Let's try to find something better with a quick greedy approach
    best_factors = [22, 21, 25, 18, 36, 64]
    best_max = 64
    
    def time_up():
        return time.time() > _DEADLINE
    
    def min_product_of_k_distinct(start, k):
        """Minimum product of k distinct integers all >= start"""
        if k <= 0:
            return 1
        if k > 20:  # Safety cap
            return float('inf')
        prod = 1
        for i in range(k):
            prod *= (start + i)
        return prod
    
    def dfs(last_val, remaining, current_max, count, current_factors, used_set):
        nonlocal best_max, best_factors
        
        if time_up():
            return
        
        if remaining == 1:
            if count >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
        
        if count >= 12:
            return
            
        # If current max is already >= best_max, we can't improve
        if current_max >= best_max:
            return
        
        # Quick check: can remaining itself be a single valid factor?
        if remaining > last_val and remaining > 12 and remaining < best_max:
            if remaining not in used_set:
                new_max = max(current_max, remaining)
                if new_max < best_max:
                    best_max = new_max
                    best_factors = current_factors + [remaining]
                return
        
        # Pruning: remaining must be > last_val
        if remaining <= last_val:
            return
        
        # The next factor must be > last_val and > 12
        min_d = max(last_val + 1, 13)
        
        # The next factor must be <= remaining and < best_max
        max_d = min(remaining, best_max - 1)
        
        if min_d > max_d:
            return
        
        # Additional pruning: if we need at least 2 more factors (d and one after),
        # then new_remaining = remaining/d must be > d, so remaining > d^2.
        # This means d < sqrt(remaining) for non-terminal factors.
        # For terminal factor (d == remaining), handled by quick check above.
        # So in the loop, if remaining > d and remaining // d <= d, skip.
        # Equivalently, d^2 >= remaining means skip (unless d == remaining).
        
        # Find range of divisors in [min_d, max_d]
        start_idx = bisect.bisect_left(valid_divs, min_d)
        end_idx = bisect.bisect_right(valid_divs, max_d)
        
        # Precompute sqrt(remaining) for pruning
        sqrt_rem = isqrt(remaining)
        
        for i in range(start_idx, end_idx):
            if time_up():
                return
                
            d = valid_divs[i]
            
            # d must divide remaining
            if remaining % d != 0:
                continue
            
            new_remaining = remaining // d
            
            # If new_remaining == 1, d is the last factor
            if new_remaining == 1:
                new_max = max(current_max, d)
                if new_max < best_max:
                    best_max = new_max
                    best_factors = current_factors + [d]
                continue
            
            # If new_remaining <= d, dead end
            if new_remaining <= d:
                continue
            
            new_max = max(current_max, d)
            
            # If new_max >= best_max, we can break because valid_divs is sorted
            if new_max >= best_max:
                break
            
            # Additional pruning: check if new_remaining can be factored into distinct
            # factors all > d and < best_max
            # Minimum number of additional factors needed: 1
            # Maximum: 12 - count - 1
            # Quick necessary condition: new_remaining must be expressible as product
            # of distinct integers in (d, best_max)
            # Lower bound: the smallest possible product of 1 factor > d is d+1
            # If new_remaining < d+1, impossible (already checked new_remaining > d)
            # For 2 factors: min product is (d+1)*(d+2)
            # If new_remaining < (d+1)*(d+2) and new_remaining != d+1 (single factor case),
            # then we need at least... actually this is complex. Let's just recurse.
            
            used_set.add(d)
            current_factors.append(d)
            
            dfs(d, new_remaining, new_max, count + 1, current_factors, used_set)
            
            current_factors.pop()
            used_set.discard(d)

    dfs(0, target, 0, 0, [], set())
    
    return best_factors
