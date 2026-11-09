import math
from functools import lru_cache

def construct():
    target = math.factorial(12)
    
    # Generate all divisors of target using prime factorization
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = set([1])
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    for p, e in zip(primes, exponents):
        new_divs = set()
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.add(val)
                val *= p
        divisors = new_divs
        
    # Filter divisors > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    n = len(valid_divs)
    
    # We want to minimize the largest factor.
    # Iterate possible maximum factor 'limit' from valid_divs in increasing order.
    for limit_idx in range(n):
        limit = valid_divs[limit_idx]
        
        @lru_cache(None)
        def dfs(start_idx, remaining):
            if remaining == 1:
                return True
            
            if start_idx >= n:
                return False
            
            # If the smallest available divisor is greater than remaining, 
            # no divisor can divide remaining (since all are > 1).
            if valid_divs[start_idx] > remaining:
                return False
                
            for i in range(start_idx, n):
                d = valid_divs[i]
                if d > limit:
                    break
                if d > remaining:
                    break
                if remaining % d == 0:
                    if dfs(i + 1, remaining // d):
                        return True
            return False
        
        if dfs(0, target):
            # Reconstruct the solution
            def find_path(start_idx, remaining, path):
                if remaining == 1:
                    return path
                
                for i in range(start_idx, n):
                    d = valid_divs[i]
                    if d > limit:
                        break
                    if d > remaining:
                        break
                    if remaining % d == 0:
                        new_path = path + [d]
                        if len(new_path) <= 12:
                            res = find_path(i + 1, remaining // d, new_path)
                            if res is not None:
                                return res
                return None
            
            result = find_path(0, target, [])
            if result:
                return result
                
    # Fallback
    return [target]
