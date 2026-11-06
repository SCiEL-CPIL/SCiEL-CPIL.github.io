import time
from math import factorial
from functools import lru_cache

def construct():
    deadline = time.time() + 80  # safety margin
    target = factorial(12)  # 479001600
    
    # 1. Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1
    divisors = set()
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if val > 12:
                        divisors.add(val)
    
    # Sort divisors in ASCENDING order
    # Ascending order allows early termination in DFS if candidate > remaining
    sorted_divs_asc = sorted(divisors)
    
    # 2. Feasibility Check
    def is_feasible(M):
        candidates = [d for d in sorted_divs_asc if d <= M]
        if not candidates:
            return False
        
        n = len(candidates)
        
        @lru_cache(maxsize=None)
        def dfs(remaining, idx, count):
            if time.time() > deadline:
                return False
            
            if remaining == 1:
                return True
            
            if count >= 12:
                return False
            
            if idx >= n:
                return False
            
            # Pruning: If the smallest available candidate is greater than remaining,
            # no candidate can divide remaining (since they are all > remaining).
            if candidates[idx] > remaining:
                return False
            
            # Option 1: Skip current candidate
            if dfs(remaining, idx + 1, count):
                return True
            
            # Option 2: Use current candidate
            if remaining % candidates[idx] == 0:
                if dfs(remaining // candidates[idx], idx + 1, count + 1):
                    return True
            
            return False
        
        result = dfs(target, 0, 0)
        dfs.cache_clear()
        return result

    # 3. Binary Search for minimal M
    # Tighten bounds based on problem constraints and inspirations
    low = 14
    high = 100
    
    # Ensure high is feasible
    while not is_feasible(high):
        high *= 2
        if high > target:
            break
            
    while low < high:
        mid = (low + high) // 2
        if is_feasible(mid):
            high = mid
        else:
            low = mid + 1
            
    optimal_M = low
    
    # 4. Reconstruct the solution
    candidates = [d for d in sorted_divs_asc if d <= optimal_M]
    n = len(candidates)
    best_factors = []
    
    def find_solution(remaining, idx, count, path):
        if time.time() > deadline:
            return False
        
        if remaining == 1:
            best_factors.extend(path)
            return True
        
        if count >= 12:
            return False
        
        if idx >= n:
            return False
        
        if candidates[idx] > remaining:
            return False
        
        # Option 1: Skip
        if find_solution(remaining, idx + 1, count, path):
            return True
        
        # Option 2: Use
        if remaining % candidates[idx] == 0:
            path.append(candidates[idx])
            if find_solution(remaining // candidates[idx], idx + 1, count + 1, path):
                return True
            path.pop()
        
        return False
    
    if not find_solution(target, 0, 0, []):
        return [target]
        
    return best_factors
