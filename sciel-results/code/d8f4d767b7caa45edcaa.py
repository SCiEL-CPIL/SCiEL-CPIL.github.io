import math
import time

def construct():
    target = math.factorial(12)
    
    # Generate all divisors of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    def get_divisors(n):
        divs = []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    all_divs = get_divisors(target)
    # Filter > 12
    valid_divs = [d for d in all_divs if d > 12]
    valid_divs.sort()
    
    # We want to find the minimum M such that there is a subset of valid_divs <= M
    # that multiplies to target.
    
    # Binary search on M
    # Low: 13
    # High: target (which is in valid_divs)
    # Actually, high can be max(valid_divs)
    
    lo = 0
    hi = len(valid_divs) - 1
    best_m = valid_divs[-1]
    best_factors = None
    
    def check(m_idx):
        # m_idx is the index in valid_divs for the max allowed factor M
        M = valid_divs[m_idx]
        candidates = valid_divs[:m_idx+1]
        
        # Memoization for this specific candidate set
        # State: (start_idx, remaining)
        # Since candidates are fixed for this call, we can use a dict or lru_cache
        # But lru_cache on a nested function is tricky if we want to clear it.
        # Let's use a dict.
        memo = {}
        
        def dfs(start_idx, remaining):
            if remaining == 1:
                return []
            
            if start_idx >= len(candidates):
                return None
            
            # Pruning: if smallest available candidate > remaining, fail
            if candidates[start_idx] > remaining:
                return None
            
            if (start_idx, remaining) in memo:
                return memo[(start_idx, remaining)]
            
            # Try each candidate from start_idx
            for i in range(start_idx, len(candidates)):
                d = candidates[i]
                if d > remaining:
                    break
                if remaining % d == 0:
                    res = dfs(i + 1, remaining // d)
                    if res is not None:
                        memo[(start_idx, remaining)] = [d] + res
                        return [d] + res
            
            memo[(start_idx, remaining)] = None
            return None
        
        return dfs(0, target)

    # Binary search
    # We want the smallest m_idx such that check(m_idx) is not None
    # Note: check(m_idx) is monotonic?
    # If a solution exists with max factor <= M, it exists with max factor <= M' > M.
    # So yes, if check(m_idx) is True, check(j) is True for all j > m_idx.
    # So we can binary search for the first True.
    
    ans_idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if check(mid) is not None:
            ans_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if ans_idx == -1:
        # Should not happen, target itself is a factor
        return [target]
        
    # Get the factors for the optimal ans_idx
    factors = check(ans_idx)
    if factors:
        return factors
    else:
        return [target]
