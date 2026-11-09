"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    # Only divisors <= target are usable
    usable = [d for d in divisors if d <= target]
    usable.sort(reverse=True)
    m = len(usable)
    
    # Precompute suffix sums for fast max-possible checks
    # suffix_sum[i] = sum of usable[i:]
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + usable[i]
    
    # For a given start index and terms_left, max possible sum is
    # sum of the largest terms_left elements from start onward
    # Precompute: max_sum[start][k] = sum of largest k elements in usable[start:]
    # But that's expensive. Instead, since usable is sorted descending,
    # the max sum of k elements from position start is sum(usable[start:start+k])
    
    def max_possible(start, k):
        """Max sum of k distinct elements from usable[start:] (they're the largest k available)"""
        if k <= 0:
            return 0
        if start + k > m:
            return -1  # not enough elements
        return sum(usable[start:start+k])
    
    def min_possible(start, k):
        """Min sum of k distinct elements from usable[start:] (smallest k available)"""
        if k <= 0:
            return 0
        if k > m - start:
            return -1
        # Smallest k elements from usable[start:] are the last k in that range
        # usable is sorted descending, so smallest are at the end
        return sum(usable[m-k:m])
    
    # Memoization: (start_idx, remaining, terms_left) -> True/False
    memo = {}
    
    def dfs(start_idx, remaining, terms_left, path):
        if time.time() > DEADLINE:
            return None
        if terms_left == 0:
            if remaining == 0:
                return list(path)
            return None
        
        # Not enough divisors left
        if m - start_idx < terms_left:
            return None
        
        # Max sum check
        max_p = max_possible(start_idx, terms_left)
        if max_p < remaining:
            return None
        
        # Min sum check  
        min_p = min_possible(start_idx, terms_left)
        if min_p > remaining:
            return None
        
        # Memo key
        key = (start_idx, remaining, terms_left)
        if key in memo:
            return None
        
        for i in range(start_idx, m):
            d = usable[i]
            if d > remaining:
                continue
            
            new_remaining = remaining - d
            new_terms_left = terms_left - 1
            
            if new_terms_left == 0:
                if new_remaining == 0:
                    path.append(d)
                    result = list(path)
                    path.pop()
                    memo[key] = True
                    return result
                continue
            
            if m - (i + 1) < new_terms_left:
                break
            
            max_after = max_possible(i + 1, new_terms_left)
            if max_after < new_remaining:
                break
            
            min_after = min_possible(i + 1, new_terms_left)
            if min_after > new_remaining:
                continue
            
            path.append(d)
            result = dfs(i + 1, new_remaining, new_terms_left, path)
            if result is not None:
                memo[key] = True
                return result
            path.pop()
        
        memo[key] = False
        return None
    
    best_result = None
    for k in range(1, 12):
        if time.time() > DEADLINE:
            break
        memo.clear()
        result = dfs(0, target, k, [])
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback to greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        best_result = chosen
    
    # Verify
    assert sum(best_result) == target
    assert len(set(best_result)) == len(best_result)
    for d in best_result:
        assert n % d == 0
    assert len(best_result) <= 256
    
    return best_result
# EVOLVE-BLOCK-END
