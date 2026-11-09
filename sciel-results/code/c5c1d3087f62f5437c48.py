import time
from math import factorial
from functools import lru_cache
import bisect

_START = time.time()
_DEADLINE = _START + 85.0

def _check_deadline():
    return time.time() < _DEADLINE

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of target > 12
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
    
    # Best solution found
    best_max = [target]
    best_factors = [[target]]
    
    def dfs(idx, current_product, current_max, factors, used):
        if not _check_deadline():
            return
        if current_product == target:
            if current_max < best_max[0]:
                best_max[0] = current_max
                best_factors[0] = list(factors)
            return
        if current_product > target:
            return
        if len(factors) >= 12:
            return
        if current_max >= best_max[0]:
            return
        
        remaining = target // current_product
        if remaining < 1:
            return
        
        # If remaining is 1, we're done (shouldn't reach here since we check current_product == target)
        
        # Pruning: the next factor must be at most remaining, and at most best_max[0]-1
        # Also, the next factor must divide remaining
        # Use binary search to find the upper bound index
        upper_val = min(remaining, best_max[0] - 1)
        # Find the rightmost index where valid_divs[i] <= upper_val
        hi = bisect.bisect_right(valid_divs, upper_val) - 1
        
        # Also, we need the next factor to be >= valid_divs[idx] (to maintain ordering)
        # And it must divide remaining
        
        for i in range(idx, hi + 1):
            if not _check_deadline():
                return
            d = valid_divs[i]
            if d in used:
                continue
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max[0]:
                continue
            
            factors.append(d)
            used.add(d)
            dfs(i + 1, current_product * d, new_max, factors, used)
            factors.pop()
            used.remove(d)
    
    # Start DFS
    dfs(0, 1, 0, [], set())
    
    return best_factors[0]
